import { useState, useEffect } from 'react'
import { 
  Send, 
  AlertTriangle, 
  CheckCircle, 
  Copy, 
  RefreshCw,
  TrendingUp,
  Clock,
  Target,
  Zap,
  Shield,
  Brain,
  Activity
} from 'lucide-react'
import LoadingSpinner from './LoadingSpinner'
import ConfidenceIndicator from './ConfidenceIndicator'
import StatCard from './StatCard'

const FakeNewsDetectorEnhanced = () => {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [examples, setExamples] = useState({ fake_examples: [], real_examples: [] })
  const [apiStatus, setApiStatus] = useState('checking')
  const [analysisHistory, setAnalysisHistory] = useState([])

  const API_BASE_URL = import.meta.env.VITE_BACKEND_URL;

  // Check API status on component mount
  useEffect(() => {
    checkApiStatus()
    loadExamples()
    loadAnalysisHistory()
  }, [])

  const checkApiStatus = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`)
      if (response.ok) {
        setApiStatus('connected')
      } else {
        setApiStatus('error')
      }
    } catch (err) {
      setApiStatus('error')
    }
  }

  const loadExamples = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/examples`)
      if (response.ok) {
        const data = await response.json()
        setExamples(data)
      }
    } catch (err) {
      console.error('Failed to load examples:', err)
    }
  }

  const loadAnalysisHistory = () => {
    const history = localStorage.getItem('analysisHistory')
    if (history) {
      setAnalysisHistory(JSON.parse(history))
    }
  }

  const saveToHistory = (analysis) => {
    const newHistory = [analysis, ...analysisHistory.slice(0, 9)] // Keep last 10
    setAnalysisHistory(newHistory)
    localStorage.setItem('analysisHistory', JSON.stringify(newHistory))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!text.trim()) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text.trim() }),
      })

      const data = await response.json()

      if (response.ok) {
        setResult(data)
        saveToHistory({
          ...data,
          originalText: text.trim(),
          analyzedAt: new Date().toISOString()
        })
      } else {
        setError(data.error || 'An error occurred')
      }
    } catch (err) {
      setError('Failed to connect to the API. Please ensure the backend is running on http://localhost:5000')
    } finally {
      setLoading(false)
    }
  }

  const handleExampleClick = (exampleText) => {
    setText(exampleText)
    setResult(null)
    setError(null)
  }

  const copyToClipboard = (textToCopy) => {
    navigator.clipboard.writeText(textToCopy)
  }

  const clearHistory = () => {
    setAnalysisHistory([])
    localStorage.removeItem('analysisHistory')
  }

  return (
    <div className="max-w-7xl mx-auto space-y-8">
      {/* API Status Banner */}
      <div className={`p-4 rounded-lg border transition-all duration-200 ${
        apiStatus === 'connected' 
          ? 'bg-green-50 border-green-200' 
          : apiStatus === 'error'
          ? 'bg-red-50 border-red-200'
          : 'bg-yellow-50 border-yellow-200'
      }`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className={`w-3 h-3 rounded-full transition-all duration-200 ${
              apiStatus === 'connected' ? 'bg-green-500 animate-pulse' : 
              apiStatus === 'error' ? 'bg-red-500' : 'bg-yellow-500 animate-pulse'
            }`}></div>
            <Activity className={`h-5 w-5 ${
              apiStatus === 'connected' ? 'text-green-600' : 
              apiStatus === 'error' ? 'text-red-600' : 'text-yellow-600'
            }`} />
            <span className="font-medium">
              {apiStatus === 'connected' ? 'AI Model Online & Ready' : 
               apiStatus === 'error' ? 'AI Model Offline' : 'Connecting to AI Model...'}
            </span>
          </div>
          <button 
            onClick={checkApiStatus}
            className="text-blue-600 hover:text-blue-800 transition-colors p-1 rounded"
            title="Refresh connection status"
          >
            <RefreshCw className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <StatCard
          title="Model Accuracy"
          value="99.5%"
          subtitle="Training Performance"
          icon={Target}
          color="green"
        />
        <StatCard
          title="Real-World Accuracy"
          value="89.5%"
          subtitle="Test Performance"
          icon={Brain}
          color="blue"
        />
        <StatCard
          title="Total Features"
          value="15,000"
          subtitle="AI Features Used"
          icon={Zap}
          color="purple"
        />
        <StatCard
          title="Analyses Today"
          value={analysisHistory.length}
          subtitle="Your Usage"
          icon={Activity}
          color="yellow"
        />
      </div>

      {/* Main Input Section */}
      <div className="bg-white rounded-xl shadow-lg p-8 border border-gray-200">
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <Shield className="h-12 w-12 text-blue-600 mr-3" />
            <Brain className="h-8 w-8 text-green-500" />
          </div>
          <h2 className="text-3xl font-bold text-gray-800 mb-3">
            AI-Powered News Analysis
          </h2>
          <p className="text-gray-600 text-lg">
            Enter any COVID-19 related news content to get instant AI analysis
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label htmlFor="news-text" className="block text-sm font-medium text-gray-700 mb-2">
              News Content to Analyze
            </label>
            <textarea
              id="news-text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste or type any COVID-19 related news text, social media post, or article here..."
              className="w-full h-40 p-4 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none transition-all duration-200"
              disabled={loading}
            />
            <div className="flex justify-between mt-2">
              <span className="text-sm text-gray-500">
                {text.length} characters
              </span>
              <span className="text-sm text-gray-500">
                {text.length >= 10 ? '✓ Ready to analyze' : 'Minimum 10 characters required'}
              </span>
            </div>
          </div>

          <button
            type="submit"
            disabled={loading || !text.trim() || text.length < 10}
            className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 text-white py-4 px-6 rounded-lg font-semibold text-lg hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2 transition-all duration-200 transform hover:scale-105 disabled:hover:scale-100"
          >
            {loading ? (
              <LoadingSpinner message="" size="small" />
            ) : (
              <>
                <Send className="h-5 w-5" />
                <span>Analyze with AI</span>
              </>
            )}
          </button>
        </form>
      </div>

      {/* Results Section */}
      {result && (
        <div className="bg-white rounded-xl shadow-lg p-8 border border-gray-200 result-card">
          <div className="text-center mb-6">
            <h3 className="text-2xl font-bold text-gray-800 mb-2">AI Analysis Complete</h3>
            <div className="flex items-center justify-center space-x-2 text-sm text-gray-500">
              <Clock className="h-4 w-4" />
              <span>Analyzed {new Date(result.timestamp).toLocaleString()}</span>
            </div>
          </div>

          {/* Main Result */}
          <div className={`text-center p-8 rounded-xl mb-6 transition-all duration-300 ${
            result.prediction === 'FAKE' 
              ? 'bg-red-50 border-2 border-red-200' 
              : 'bg-green-50 border-2 border-green-200'
          }`}>
            <div className="flex items-center justify-center mb-4">
              {result.prediction === 'FAKE' ? (
                <AlertTriangle className="h-16 w-16 text-red-500 animate-bounce" />
              ) : (
                <CheckCircle className="h-16 w-16 text-green-500" />
              )}
            </div>
            <h4 className={`text-4xl font-bold mb-2 ${
              result.prediction === 'FAKE' ? 'text-red-600' : 'text-green-600'
            }`}>
              {result.prediction === 'FAKE' ? 'FAKE NEWS DETECTED' : 'LEGITIMATE NEWS'}
            </h4>
            <p className="text-gray-600 text-lg">
              {result.prediction === 'FAKE' 
                ? 'This content likely contains misinformation or false claims'
                : 'This content appears to be factual and legitimate'
              }
            </p>
          </div>

          {/* Confidence Indicator */}
          <ConfidenceIndicator 
            confidence={result.confidence}
            prediction={result.prediction}
            className="mb-6"
          />

          {/* Detailed Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <StatCard
              title="Fake Probability"
              value={`${result.fake_probability}%`}
              icon={TrendingUp}
              color={result.fake_probability > 50 ? "red" : "green"}
            />
            <StatCard
              title="Real Probability" 
              value={`${result.real_probability}%`}
              icon={CheckCircle}
              color={result.real_probability > 50 ? "green" : "red"}
            />
            <StatCard
              title="Model Used"
              value={result.model_type}
              subtitle="AI Engine"
              icon={Brain}
              color="purple"
            />
          </div>

          {/* Processed Text */}
          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium text-gray-700">AI Processed Text:</span>
              <button
                onClick={() => copyToClipboard(result.processed_text)}
                className="text-blue-600 hover:text-blue-800 transition-colors"
                title="Copy processed text"
              >
                <Copy className="h-4 w-4" />
              </button>
            </div>
            <p className="text-gray-600 text-sm">{result.processed_text}</p>
          </div>
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="bg-white rounded-xl shadow-lg p-8 border border-gray-200">
          <LoadingSpinner message="AI is analyzing your content..." size="large" />
        </div>
      )}

      {/* Error Section */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <div className="flex items-center space-x-2">
            <AlertTriangle className="h-5 w-5 text-red-500" />
            <h4 className="font-medium text-red-800">Analysis Error</h4>
          </div>
          <p className="text-red-700 mt-2">{error}</p>
          <button
            onClick={() => setError(null)}
            className="mt-3 text-sm text-red-600 hover:text-red-800 underline"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Examples and History Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Fake Examples */}
        <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
          <h3 className="text-xl font-bold text-red-600 mb-4 flex items-center">
            <AlertTriangle className="h-5 w-5 mr-2" />
            Fake News Examples
          </h3>
          <div className="space-y-3">
            {examples.fake_examples.slice(0, 3).map((example, index) => (
              <button
                key={index}
                onClick={() => handleExampleClick(example)}
                className="w-full text-left p-3 bg-red-50 border border-red-200 rounded-lg hover:bg-red-100 transition-colors text-sm"
              >
                <p className="text-gray-700">{example}</p>
              </button>
            ))}
          </div>
        </div>

        {/* Real Examples */}
        <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
          <h3 className="text-xl font-bold text-green-600 mb-4 flex items-center">
            <CheckCircle className="h-5 w-5 mr-2" />
            Real News Examples
          </h3>
          <div className="space-y-3">
            {examples.real_examples.slice(0, 3).map((example, index) => (
              <button
                key={index}
                onClick={() => handleExampleClick(example)}
                className="w-full text-left p-3 bg-green-50 border border-green-200 rounded-lg hover:bg-green-100 transition-colors text-sm"
              >
                <p className="text-gray-700">{example}</p>
              </button>
            ))}
          </div>
        </div>

        {/* Analysis History */}
        <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold text-purple-600 flex items-center">
              <Clock className="h-5 w-5 mr-2" />
              Recent Analyses
            </h3>
            {analysisHistory.length > 0 && (
              <button
                onClick={clearHistory}
                className="text-sm text-gray-500 hover:text-gray-700 underline"
              >
                Clear
              </button>
            )}
          </div>
          <div className="space-y-3">
            {analysisHistory.length === 0 ? (
              <p className="text-gray-500 text-sm text-center py-4">
                No analyses yet. Try analyzing some content!
              </p>
            ) : (
              analysisHistory.slice(0, 3).map((item, index) => (
                <div
                  key={index}
                  className={`p-3 rounded-lg border text-sm ${
                    item.prediction === 'FAKE' 
                      ? 'bg-red-50 border-red-200' 
                      : 'bg-green-50 border-green-200'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className={`font-medium ${
                      item.prediction === 'FAKE' ? 'text-red-600' : 'text-green-600'
                    }`}>
                      {item.prediction}
                    </span>
                    <span className="text-xs text-gray-500">
                      {new Date(item.analyzedAt).toLocaleDateString()}
                    </span>
                  </div>
                  <p className="text-gray-600 text-xs truncate">
                    {item.originalText.substring(0, 60)}...
                  </p>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default FakeNewsDetectorEnhanced