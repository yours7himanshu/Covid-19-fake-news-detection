# COVID-19 Fake News Detection Frontend Components

This directory contains all the React components for the COVID-19 Fake News Detection system.

## Components Overview

### Core Components
- **`FakeNewsDetectorEnhanced.jsx`** - Main application component with AI analysis functionality
- **`Header.jsx`** - Application header with branding and statistics
- **`Footer.jsx`** - Detailed footer with model information and architecture details

### UI Components
- **`LoadingSpinner.jsx`** - Reusable loading animation component
- **`ConfidenceIndicator.jsx`** - Visual confidence level indicator with progress bar
- **`StatCard.jsx`** - Reusable statistics display card component
- **`Toast.jsx`** - Notification system with success/error/warning messages

### Legacy Components
- **`FakeNewsDetector.jsx`** - Original detector component (replaced by Enhanced version)

## Features

### Enhanced Fake News Detector
- ✅ Real-time API status monitoring
- ✅ Interactive examples (fake and real news)
- ✅ Analysis history with local storage
- ✅ Advanced confidence visualization
- ✅ Responsive design for all screen sizes
- ✅ Accessibility features (focus states, ARIA labels)
- ✅ Copy to clipboard functionality
- ✅ Model performance statistics

### UI/UX Features
- 🎨 Beautiful gradient backgrounds
- 📱 Fully responsive design
- ♿ Accessible components
- 🌟 Smooth animations and transitions
- 🎯 Interactive hover states
- 📊 Visual data representation
- 🔄 Loading states and error handling

## API Integration

The frontend communicates with a Flask backend API at `http://localhost:5000` with the following endpoints:

- `GET /health` - Check API status
- `POST /predict` - Analyze news content
- `GET /examples` - Get sample news content
- `GET /model_info` - Get model information

## Usage

To use these components in your application:

```jsx
import FakeNewsDetectorEnhanced from './components/FakeNewsDetectorEnhanced'
import Header from './components/Header'
import Footer from './components/Footer'

function App() {
  return (
    <div>
      <Header />
      <FakeNewsDetectorEnhanced />
      <Footer />
    </div>
  )
}
```

## Development

All components are built with:
- React 18+ with hooks
- Tailwind CSS for styling
- Lucide React for icons
- Modern ES6+ JavaScript

## Performance

- Components use React.memo where appropriate
- Local storage for analysis history
- Debounced API calls
- Efficient re-rendering with proper key props