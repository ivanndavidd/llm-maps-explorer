# backend/main.py
"""
Complete Backend for LLM Maps Integration - Fixed Version
FastAPI + Google Maps New Places API + Open WebUI Integration

Key Features:
- Google Maps New Places API integration (working)
- Open WebUI (LLM) integration with fallback responses
- Redis caching for performance optimization
- Rate limiting and security features
- Comprehensive error handling and logging
- Input validation with Pydantic models
- Full CORS support for frontend integration
"""

import os
import json
import logging
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

# FastAPI and related imports
from fastapi import FastAPI, HTTPException, Depends, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Pydantic for data validation
from pydantic import BaseModel, Field, field_validator

# External libraries
import requests
from dotenv import load_dotenv

# Optional dependencies with graceful fallback
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    RATE_LIMIT_AVAILABLE = True
except ImportError:
    RATE_LIMIT_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logging
def setup_logging():
    """Setup logging configuration with Windows compatibility"""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    # Set external library log levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("requests").setLevel(logging.WARNING)

setup_logging()
logger = logging.getLogger(__name__)

# Configuration
class Config:
    """Application configuration"""
    
    # Required settings
    GOOGLE_MAPS_API_KEY: str = os.getenv("GOOGLE_MAPS_API_KEY")
    API_SECRET_KEY: str = os.getenv("API_SECRET_KEY", "my-development-secret-key")
    
    # Open WebUI settings
    OPENWEBUI_API_KEY: str = os.getenv("OPENWEBUI_API_KEY", "")
    OPENWEBUI_BASE_URL: str = os.getenv("OPENWEBUI_BASE_URL", "http://localhost:3000")
    OPENWEBUI_MODEL: str = os.getenv("OPENWEBUI_MODEL", "gemma3:latest")
    OPENWEBUI_TIMEOUT: int = int(os.getenv("OPENWEBUI_TIMEOUT", "30"))
    
    # Redis settings
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    CACHE_TTL: int = int(os.getenv("CACHE_TTL", "3600"))
    
    # API settings
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = os.getenv(
        "ALLOWED_ORIGINS", 
        "http://localhost:3000,http://localhost:8080,http://127.0.0.1:5500"
    ).split(",")
    
    # Search settings
    MAX_PLACES: int = int(os.getenv("MAX_PLACES", "15"))
    DEFAULT_RADIUS: int = int(os.getenv("DEFAULT_RADIUS", "5000"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    
    @classmethod
    def validate_required_settings(cls):
        """Validate required environment variables"""
        if not cls.GOOGLE_MAPS_API_KEY:
            error_msg = "GOOGLE_MAPS_API_KEY environment variable is required"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Configuration validation successful")

# Initialize configuration
config = Config()

# Cache Manager
class CacheManager:
    """Redis caching operations manager"""
    
    def __init__(self):
        self.client = None
        self.available = False
        
    async def initialize(self):
        """Initialize Redis connection if available"""
        if not REDIS_AVAILABLE:
            logger.warning("Redis not installed. Caching disabled.")
            return
            
        try:
            redis_kwargs = {
                "host": config.REDIS_HOST,
                "port": config.REDIS_PORT,
                "db": config.REDIS_DB,
                "decode_responses": True,
                "socket_connect_timeout": 5,
                "socket_timeout": 5,
                "retry_on_timeout": True
            }
            
            if config.REDIS_PASSWORD:
                redis_kwargs["password"] = config.REDIS_PASSWORD
                
            self.client = redis.Redis(**redis_kwargs)
            await asyncio.to_thread(self.client.ping)
            self.available = True
            logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            logger.warning(f"Redis initialization failed: {e}. Caching disabled.")
            self.available = False
    
    def generate_cache_key(self, query: str, location: str, radius: int, place_type: str) -> str:
        """Generate unique cache key"""
        key_data = f"{query.lower().strip()}:{location.lower().strip()}:{radius}:{place_type}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"maps_search:{key_hash}"
    
    async def get(self, key: str) -> Optional[Dict]:
        """Retrieve cached data"""
        if not self.available:
            return None
            
        try:
            cached_data = await asyncio.to_thread(self.client.get, key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.error(f"Cache retrieval error for key {key}: {e}")
        
        return None
    
    async def set(self, key: str, data: Dict, ttl: int = None) -> bool:
        """Store data in cache"""
        if not self.available:
            return False
            
        try:
            ttl = ttl or config.CACHE_TTL
            await asyncio.to_thread(
                self.client.setex,
                key,
                ttl,
                json.dumps(data, default=str, ensure_ascii=False)
            )
            return True
        except Exception as e:
            logger.error(f"Cache storage error for key {key}: {e}")
            return False

# Initialize cache manager
cache_manager = CacheManager()

# LLM Service - Version tanpa timeout, menunggu sampai selesai
class LLMService:
    """Service for Open WebUI LLM integration tanpa timeout"""
    
    @staticmethod
    async def query_llm(prompt: str, context: Dict = None) -> str:
        """Query LLM tanpa timeout, menunggu response sampai selesai"""
        try:
            if not config.OPENWEBUI_API_KEY:
                logger.warning("OpenWebUI API key not configured, using fallback response")
                return LLMService._generate_fallback_response(prompt, context)
            
            # System prompt
            system_prompt = """You are a knowledgeable local search assistant. Provide helpful recommendations for places and locations.

Your responses should be:
- Informative and contextual (2-3 sentences)
- Include practical tips about the type of places
- Be conversational and friendly
- Focus on what makes good choices for their search

Example: "Great coffee shops focus on quality beans and cozy atmosphere. Look for places with recent positive reviews and good ratings for the best experience."
"""

            # Create contextual prompt
            enhanced_prompt = prompt
            if context:
                location_context = f" in {context.get('location', 'the area')}" if context.get('location') else ""
                enhanced_prompt = f"User is searching for: {prompt}{location_context}"
            
            headers = {
                "Authorization": f"Bearer {config.OPENWEBUI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": config.OPENWEBUI_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": enhanced_prompt}
                ],
                "stream": False,
                "temperature": 0.7,
                "max_tokens": 150
            }
            
            api_url = f"{config.OPENWEBUI_BASE_URL}/api/chat/completions"
            
            logger.info(f"Calling OpenWebUI at: {api_url}")
            logger.info(f"Using model: {config.OPENWEBUI_MODEL}")
            logger.info("Waiting for LLM response (no timeout)...")
            
            # Request tanpa timeout - menunggu sampai selesai
            response = await asyncio.to_thread(
                requests.post,
                api_url,
                headers=headers,
                json=payload
                # Tidak ada timeout parameter
            )
            
            logger.info(f"OpenWebUI response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Handle response format
                if "choices" in result and len(result["choices"]) > 0:
                    llm_response = result["choices"][0]["message"]["content"].strip()
                    logger.info("LLM response generated successfully")
                    return llm_response
                else:
                    logger.warning(f"Unexpected response format: {result}")
                    return LLMService._generate_fallback_response(prompt, context)
            
            elif response.status_code == 401:
                logger.error("OpenWebUI returned 401 - Check your API key")
                return LLMService._generate_fallback_response(prompt, context)
            else:
                logger.warning(f"LLM API returned status {response.status_code}: {response.text}")
                return LLMService._generate_fallback_response(prompt, context)
                
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            return LLMService._generate_fallback_response(prompt, context)
    
    @staticmethod
    def _generate_fallback_response(prompt: str, context: Dict = None) -> str:
        """Generate fallback response when LLM unavailable"""
        location = context.get('location', 'your area') if context else 'your area'
        
        # Enhanced fallback responses
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['restaurant', 'food', 'eat', 'dining', 'makan']):
            return f"Look for restaurants in {location} with high ratings and recent reviews. Check opening hours and price ranges to find the perfect dining experience for your needs."
        elif any(word in prompt_lower for word in ['coffee', 'cafe', 'kopi']):
            return f"Great coffee shops in {location} focus on quality beans and cozy atmosphere. Look for places with good ratings and comfortable seating for the best coffee experience."
        elif any(word in prompt_lower for word in ['gas', 'fuel', 'petrol', 'spbu']):
            return f"Found gas stations in {location}. Compare fuel prices and check for additional services like convenience stores or car wash facilities."
        elif any(word in prompt_lower for word in ['hospital', 'medical', 'doctor', 'rumah sakit']):
            return f"Medical facilities in {location} are listed with contact information. For emergencies, always call emergency services first. Check operating hours for routine visits."
        elif any(word in prompt_lower for word in ['hotel', 'lodging', 'stay', 'penginapan']):
            return f"Accommodation options in {location} vary by price and amenities. Compare ratings, location convenience, and guest reviews to find the best fit for your stay."
        elif any(word in prompt_lower for word in ['shopping', 'mall', 'store', 'belanja']):
            return f"Shopping destinations in {location} offer various options. Check operating hours, available stores, and parking facilities before visiting."
        else:
            return f"Found several options for '{prompt}' in {location}. Compare ratings, reviews, and locations to choose the best option for your specific needs."

# Google Maps Service with New Places API
class GoogleMapsService:
    """Service for Google Maps New Places API operations"""
    
    def __init__(self):
        self.api_key = config.GOOGLE_MAPS_API_KEY
        self.base_url = "https://places.googleapis.com/v1"
    
    async def search_places(self, query: str, location: str = None, radius: int = 5000, place_type: str = None) -> List[Dict]:
        """Search places using Google Maps New Places API"""
        try:
            # Prepare headers for New Places API
            headers = {
                "Content-Type": "application/json",
                "X-Goog-Api-Key": self.api_key,
                "X-Goog-FieldMask": "places.id,places.displayName,places.formattedAddress,places.location,places.rating,places.priceLevel,places.types,places.photos,places.regularOpeningHours,places.internationalPhoneNumber,places.websiteUri"
            }
            
            # Construct search text
            search_text = query
            if location:
                search_text += f" in {location}"
            
            payload = {
                "textQuery": search_text,
                "maxResultCount": config.MAX_PLACES
            }
            
            logger.info(f"Searching places with New Places API: {search_text}")
            
            # Make API request
            response = await asyncio.to_thread(
                requests.post,
                f"{self.base_url}/places:searchText",
                headers=headers,
                json=payload,
                timeout=config.REQUEST_TIMEOUT
            )
            
            if response.status_code != 200:
                raise Exception(f"Places API returned status {response.status_code}: {response.text}")
            
            result = response.json()
            places = result.get('places', [])
            
            # Convert to internal format
            enhanced_places = []
            for place in places:
                enhanced_place = self._convert_new_api_format(place)
                enhanced_places.append(enhanced_place)
            
            logger.info(f"Found {len(enhanced_places)} places")
            return enhanced_places
            
        except Exception as e:
            logger.error(f"Google Maps search failed: {e}")
            raise HTTPException(
                status_code=500, 
                detail=f"Places search failed: {str(e)}"
            )
    
    def _convert_new_api_format(self, place: Dict) -> Dict:
        """Convert New Places API format to internal format"""
        display_name = place.get('displayName', {})
        location = place.get('location', {})
        opening_hours = place.get('regularOpeningHours', {})
        
        # Fix price level conversion
        price_level = place.get('priceLevel')
        if price_level:
            # Convert string enum to integer
            price_mapping = {
                'PRICE_LEVEL_FREE': 0,
                'PRICE_LEVEL_INEXPENSIVE': 1,
                'PRICE_LEVEL_MODERATE': 2,
                'PRICE_LEVEL_EXPENSIVE': 3,
                'PRICE_LEVEL_VERY_EXPENSIVE': 4
            }
            price_level = price_mapping.get(price_level, None)
        
        # Process photos
        photos = []
        if 'photos' in place:
            for photo in place['photos'][:3]:
                photo_name = photo.get('name', '')
                if photo_name:
                    photo_url = f"https://places.googleapis.com/v1/{photo_name}/media?maxWidthPx=400&key={self.api_key}"
                    photos.append(photo_url)
        
        return {
            'place_id': place.get('id', ''),
            'name': display_name.get('text', 'Unknown'),
            'address': place.get('formattedAddress', 'Address not available'),
            'rating': place.get('rating'),
            'price_level': price_level,  # Now properly converted to int
            'location': {
                'lat': location.get('latitude', 0),
                'lng': location.get('longitude', 0)
            },
            'types': place.get('types', []),
            'photos': photos,
            'opening_hours': self._format_opening_hours(opening_hours),
            'phone': place.get('internationalPhoneNumber'),
            'website': place.get('websiteUri'),
            'geometry': {
                'location': {
                    'lat': location.get('latitude', 0),
                    'lng': location.get('longitude', 0)
                }
            }
        }
    
    def _format_opening_hours(self, opening_hours: Dict) -> Optional[Dict]:
        """Format opening hours from New Places API"""
        if not opening_hours:
            return None
        
        return {
            'open_now': opening_hours.get('openNow', False),
            'weekday_text': opening_hours.get('weekdayDescriptions', [])
        }
    
    def calculate_map_center(self, places: List[Dict], default_location: str = None) -> Dict[str, float]:
        """Calculate center point for map display"""
        if places:
            valid_places = [p for p in places if p.get('location', {}).get('lat') and p.get('location', {}).get('lng')]
            
            if valid_places:
                total_lat = sum(place['location']['lat'] for place in valid_places)
                total_lng = sum(place['location']['lng'] for place in valid_places)
                return {
                    'lat': total_lat / len(valid_places),
                    'lng': total_lng / len(valid_places)
                }
        
        # Default to Jakarta coordinates
        return {'lat': -6.2088, 'lng': 106.8456}

# Initialize Google Maps service
maps_service = GoogleMapsService()

# Pydantic models
class SearchRequest(BaseModel):
    """Request model for place search"""
    query: str = Field(..., min_length=1, max_length=200, description="Search query for places")
    location: Optional[str] = Field(None, max_length=200, description="Location context")
    radius: Optional[int] = Field(5000, ge=100, le=50000, description="Search radius in meters")
    type: Optional[str] = Field(None, max_length=50, description="Place type filter")
    
    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()
    
    @field_validator('location')
    @classmethod
    def validate_location(cls, v):
        return v.strip() if v and v.strip() else None

class PlaceResponse(BaseModel):
    """Response model for place information"""
    place_id: str
    name: str
    address: str
    rating: Optional[float] = None
    price_level: Optional[int] = None
    location: Dict[str, float]
    types: List[str] = []
    photos: List[str] = []
    opening_hours: Optional[Dict] = None
    phone: Optional[str] = None
    website: Optional[str] = None

class SearchResponse(BaseModel):
    """Response model for search results"""
    llm_response: str
    places: List[PlaceResponse]
    map_center: Dict[str, float]
    search_query: str
    total_results: int
    processing_time_ms: float
    cached: bool = False

class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: str
    services: Dict[str, str]
    version: str = "1.0.0"

# Security
security = HTTPBearer()

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify API key for authentication"""
    if credentials.credentials != config.API_SECRET_KEY:
        logger.warning("Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

# Rate limiting
if RATE_LIMIT_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown"""
    # Startup
    logger.info("Starting LLM Maps Integration API...")
    
    config.validate_required_settings()
    await cache_manager.initialize()
    
    # Test Google Maps New Places API
    try:
        test_headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": config.GOOGLE_MAPS_API_KEY
        }
        test_response = await asyncio.to_thread(
            requests.post,
            "https://places.googleapis.com/v1/places:searchText",
            headers=test_headers,
            json={"textQuery": "test"},
            timeout=10
        )
        if test_response.status_code in [200, 400]:  # 400 is OK for test query
            logger.info("Google Maps New Places API connection verified")
        else:
            logger.warning(f"Google Maps API test returned status: {test_response.status_code}")
    except Exception as e:
        logger.error(f"Google Maps API test failed: {e}")
    
    logger.info("API startup completed successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down API...")

# Initialize FastAPI
app = FastAPI(
    title="LLM Maps Integration API",
    description="AI-powered location search with Google Maps integration",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if config.DEBUG else None,
    redoc_url="/redoc" if config.DEBUG else None
)

# Add rate limiting
if RATE_LIMIT_AVAILABLE:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    start_time = datetime.now()
    
    logger.info(f"Request: {request.method} {request.url.path} - Client: {request.client.host}")
    
    response = await call_next(request)
    
    process_time = (datetime.now() - start_time).total_seconds() * 1000
    logger.info(f"Response: {response.status_code} - Time: {process_time:.2f}ms")
    
    return response

# API Routes
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "LLM Maps Integration API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "search": "/api/search",
            "health": "/api/health",
            "docs": "/docs" if config.DEBUG else "disabled"
        }
    }

@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    services_status = {
        "google_maps": "connected",
        "cache": "connected" if cache_manager.available else "disconnected",
        "llm": "configured" if config.OPENWEBUI_API_KEY else "not_configured"
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        services=services_status
    )

@app.post("/api/search", response_model=SearchResponse, tags=["Search"])
async def search_places(
    search_request: SearchRequest,
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """Search for places using AI and Google Maps"""
    start_time = datetime.now()
    
    try:
        # Check cache
        cache_key = cache_manager.generate_cache_key(
            search_request.query,
            search_request.location or "",
            search_request.radius or config.DEFAULT_RADIUS,
            search_request.type or ""
        )
        
        cached_result = await cache_manager.get(cache_key)
        if cached_result:
            logger.info(f"Cache hit for search: {search_request.query}")
            cached_result["processing_time_ms"] = (datetime.now() - start_time).total_seconds() * 1000
            cached_result["cached"] = True
            return SearchResponse(**cached_result)
        
        # Concurrent LLM and Maps API calls
        llm_context = {
            "location": search_request.location,
            "type": search_request.type
        }
        
        llm_task = LLMService.query_llm(search_request.query, llm_context)
        maps_task = maps_service.search_places(
            query=search_request.query,
            location=search_request.location,
            radius=search_request.radius or config.DEFAULT_RADIUS,
            place_type=search_request.type
        )
        
        llm_response, places_data = await asyncio.gather(llm_task, maps_task)
        
        # Calculate map center
        map_center = maps_service.calculate_map_center(places_data, search_request.location)
        
        # Prepare response
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        response_data = {
            "llm_response": llm_response,
            "places": [PlaceResponse(**place) for place in places_data],
            "map_center": map_center,
            "search_query": search_request.query,
            "total_results": len(places_data),
            "processing_time_ms": processing_time,
            "cached": False
        }
        
        # Cache result
        await cache_manager.set(cache_key, response_data)
        
        logger.info(f"Search completed: {search_request.query} - {len(places_data)} results in {processing_time:.2f}ms")
        
        return SearchResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.get("/api/place/{place_id}", tags=["Places"])
async def get_place_details(
    place_id: str,
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """Get detailed place information"""
    try:
        # Use New Places API for place details
        headers = {
            "Content-Type": "application/json",
            "X-Goog-Api-Key": config.GOOGLE_MAPS_API_KEY,
            "X-Goog-FieldMask": "id,displayName,formattedAddress,internationalPhoneNumber,websiteUri,rating,reviews,regularOpeningHours,location,photos,priceLevel,types"
        }
        
        response = await asyncio.to_thread(
            requests.get,
            f"https://places.googleapis.com/v1/places/{place_id}",
            headers=headers,
            timeout=config.REQUEST_TIMEOUT
        )
        
        if response.status_code == 200:
            return {"place_details": response.json()}
        else:
            raise HTTPException(status_code=response.status_code, detail="Failed to get place details")
        
    except Exception as e:
        logger.error(f"Place details error for {place_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get place details: {str(e)}"
        )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Handle 404 errors"""
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "path": str(request.url.path)}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP {exc.status_code} error for {request.url.path}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

# Development server
if __name__ == "__main__":
    import uvicorn
    
    # Startup information
    logger.info("="*50)
    logger.info("LLM Maps Integration API Starting...")
    logger.info(f"Debug mode: {config.DEBUG}")
    logger.info(f"Host: {config.HOST}:{config.PORT}")
    logger.info(f"Google Maps API: {'CONFIGURED' if config.GOOGLE_MAPS_API_KEY else 'MISSING'}")
    logger.info(f"Open WebUI: {'CONFIGURED' if config.OPENWEBUI_API_KEY else 'NOT_CONFIGURED'}")
    logger.info(f"Redis Cache: {'AVAILABLE' if REDIS_AVAILABLE else 'NOT_AVAILABLE'}")
    logger.info(f"Rate Limiting: {'AVAILABLE' if RATE_LIMIT_AVAILABLE else 'NOT_AVAILABLE'}")
    logger.info("="*50)
    
    # Run server
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info" if config.DEBUG else "warning"
    )