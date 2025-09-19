# AI Maps Explorer

AI-powered location search with Google Maps integration and LLM recommendations.

## Features

- **AI-Powered Recommendations**: Get intelligent suggestions for places using OpenWebUI with Ollama
- **Interactive Maps**: Google Maps integration with markers and place details
- **Smart Search**: Search for restaurants, cafes, gas stations, and more
- **Location Detection**: Automatic location detection with GPS
- **Real-time Results**: Fast search with caching support
- **Modern UI**: Clean, responsive interface with animations

## Tech Stack

### Backend

- **FastAPI** - Modern Python web framework
- **Google Maps New Places API** - Location data and search
- **OpenWebUI + Ollama** - Local LLM integration (Gemma3)
- **Pydantic** - Data validation
- **Python 3.10+**

### Frontend

- **Vanilla JavaScript** - No frameworks, pure JS
- **Google Maps JavaScript API** - Interactive maps
- **Modern CSS** - Custom properties, flexbox, grid
- **Responsive Design** - Mobile-first approach

## Prerequisites

1. **Python 3.10+**
2. **Google Maps API Key** - Get from [Google Cloud Console](https://console.cloud.google.com/)
3. **OpenWebUI + Ollama** - For AI recommendations

### Setting up Ollama and OpenWebUI

1. **Install Ollama**:

   ```bash
   # Windows/Mac/Linux - Download from https://ollama.ai
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Pull Gemma3 model**:

   ```bash
   ollama pull gemma3:latest
   ```

3. **Install OpenWebUI**:

   ```bash
   # Using Docker (recommended)
   docker run -d --name open-webui -p 3000:8080 \
     -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
     -v open-webui:/app/backend/data \
     ghcr.io/open-webui/open-webui:main

   # Or using pip
   pip install open-webui
   open-webui serve --port 3000
   ```

4. **Get OpenWebUI API Key**:
   - Open http://localhost:3000
   - Sign up/Login
   - Go to Settings → Account → API Keys
   - Generate new API key

## Installation

1. **Clone the repository**:

   ```bash
   git clone
   cd
   ```

2. **Create Python virtual environment**:

   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate

   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Python dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Create environment file**:

   ```bash
   cp .env.example .env
   ```

5. **Configure `.env` file**:

   ```env
    GOOGLE_MAPS_API_KEY=your_actual_google_maps_api_key_here
    API_SECRET_KEY=my-secret-key-for-development

    # OpenWebUI settings (sesuai setup Anda)
    OPENWEBUI_BASE_URL=http://localhost:3000
    OPENWEBUI_MODEL=gemma3:latest
    OPENWEBUI_API_KEY=your_openwebui_api_key_here

    # Basic settings
    DEBUG=true
    HOST=0.0.0.0
    PORT=8000
   ```

## Getting API Keys

### Google Maps API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing
3. Enable these APIs:
   - **Maps JavaScript API**
   - **Places API (New)**
4. Go to **Credentials** → **Create Credentials** → **API Key**
5. Restrict the API key to your domain for production

### OpenWebUI API Key

1. Start OpenWebUI: `http://localhost:3000`
2. Create account or login
3. Go to **Settings** (top right)
4. Navigate to **Account** → **API Keys**
5. Click **Generate API Key**
6. Copy the generated JWT token

## Usage

### 1. Start the Backend Server

```bash
# Development mode (with auto-reload)
python main.py
```

The API will be available at:

- **Main API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

### 2. Open the Frontend

Open `index.html` in your browser or serve it with a local server:

```bash
    start index.html
```

### 3. Testing the Setup

1. **Test Backend Health**:
   ```bash
   curl http://localhost:8000/api/health
   ```

## API Endpoints

### Search Places

```http
POST /api/search
Authorization: Bearer your-api-secret-key
Content-Type: application/json

{
  "query": "coffee shops",
  "location": "Jakarta",
  "radius": 5000,
  "type": "cafe"
}
```

````

### Place Details
```http
GET /api/place/{place_id}
Authorization: Bearer your-api-secret-key
````

## Configuration

### Environment Variables

| Variable              | Description                | Default                 | Required |
| --------------------- | -------------------------- | ----------------------- | -------- |
| `GOOGLE_MAPS_API_KEY` | Google Maps API key        | -                       | ✅       |
| `API_SECRET_KEY`      | Backend API authentication | -                       | ✅       |
| `OPENWEBUI_API_KEY`   | OpenWebUI API key          | -                       | ✅       |
| `OPENWEBUI_BASE_URL`  | OpenWebUI server URL       | `http://localhost:3000` |          |
| `OPENWEBUI_MODEL`     | LLM model name             | `gemma3:latest`         |          |
| `DEBUG`               | Enable debug mode          | `false`                 |          |
| `HOST`                | Server host                | `0.0.0.0`               |          |
| `PORT`                | Server port                | `8000`                  |          |

### Frontend Configuration

The frontend automatically loads configuration from the backend at `/api/config`. You can also manually configure it by editing the `CONFIG` object in `index.html`:

```javascript
const CONFIG = {
  API_BASE_URL: "http://localhost:8000",
  API_SECRET_KEY: "development-key-123",
  GOOGLE_MAPS_API_KEY: "your-maps-api-key",
  DEFAULT_LOCATION: "Jakarta",
  DEFAULT_COORDINATES: { lat: -6.2088, lng: 106.8456 },
  MAP_ZOOM: 12,
  DETAILED_ZOOM: 16,
};
```

## Deployment

### Production Environment

1. **Create production virtual environment**:

   ```bash
   python -m venv venv-prod
   source venv-prod/bin/activate  # Linux/macOS
   # or venv-prod\Scripts\activate  # Windows
   ```

2. **Install production dependencies**:

   ```bash
   pip install -r requirements.txt
   pip install gunicorn
   ```

3. **Update `.env` for production**:

   ```env
   DEBUG=false
   HOST=0.0.0.0
   PORT=8000
   API_SECRET_KEY=your-super-secure-production-key
   ```

4. **Run with Gunicorn**:
   ```bash
   gunicorn main:app -w 4 -k uvicorn.workers.UnicornWorker --bind 0.0.0.0:8000
   ```

### Common Issues

1. **"GOOGLE_MAPS_API_KEY environment variable is required"**

   - Make sure `.env` file exists and has the correct API key
   - Verify the API key has the required permissions

2. **LLM API returns 401**

   - Check OpenWebUI is running on port 3000
   - Verify the API key is correct and not expired
   - Test OpenWebUI manually at http://localhost:3000

3. **"Places search failed"**

   - Verify Google Maps API key has Places API enabled
   - Check API quota limits in Google Cloud Console
   - Ensure billing is enabled for your Google Cloud project

4. **Frontend shows "Failed to load configuration"**
   - Make sure backend server is running
   - Check CORS settings allow frontend origin
   - Verify API_SECRET_KEY matches between frontend and backend

### Debug Mode

Enable debug logging by setting `DEBUG=true` in `.env`:

```bash
DEBUG=true
LOG_LEVEL=DEBUG
```

Check logs in `logs/app.log` for detailed information.

### Testing Individual Components

1. **Test Ollama**:

   ```bash
   curl http://localhost:11434/api/generate \
     -d '{"model": "gemma2:latest", "prompt": "Hello", "stream": false}'
   ```

2. **Test Google Maps API**:
   ```bash
   curl "https://places.googleapis.com/v1/places:searchText" \
     -H "Content-Type: application/json" \
     -H "X-Goog-Api-Key: YOUR_API_KEY" \
     -d '{"textQuery": "restaurants in Jakarta"}'
   ```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [Google Maps Platform](https://developers.google.com/maps) - Location services
- [OpenWebUI](https://github.com/open-webui/open-webui) - LLM interface
- [Ollama](https://ollama.ai/) - Local LLM runtime

## Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#troubleshooting) section
2. Create a new issue with detailed information about your problem

---
