import { useState, useEffect } from 'react'
import { Send, AlertTriangle, CheckCircle, RefreshCw, Brain, Shield, Zap, Activity } from 'lucide-react'

const FakeNewsDetector = () => {
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [apiStatus, setApiStatus] = useState('checking')

  const API_BASE_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5000';

  // Check API status on component mount
  useEffect(() => {
    checkApiStatus()
    const interval = setInterval(checkApiStatus, 30000) // Check every 30 seconds
    return () => clearInterval(interval)
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

  const analyzeText = async () => {
    if (!text.trim()) {
      setError('Please enter some text to analyze')
      return
    }

    if (text.trim().length < 10) {
      setError('Please enter at least 10 characters for better analysis')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text.trim() })
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(`Analysis failed: ${err.message}. Please ensure the backend is running.`)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && e.ctrlKey && !loading) {
      analyzeText()
    }
  }

  const clearAll = () => {
    setText('')
    setResult(null)
    setError(null)
  }

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'from-green-500 to-emerald-600'
    if (confidence >= 0.6) return 'from-yellow-500 to-orange-500'
    return 'from-red-500 to-pink-600'
  }

  const getConfidenceLabel = (confidence) => {
    if (confidence >= 0.9) return 'Very High Confidence'
    if (confidence >= 0.7) return 'High Confidence'
    if (confidence >= 0.5) return 'Medium Confidence'
    return 'Low Confidence'
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="relative">
              <Brain className="w-12 h-12 text-blue-600" />
              <div className="absolute -top-1 -right-1 w-6 h-6 bg-gradient-to-r from-purple-500 to-pink-500 rounded-full flex items-center justify-center">
                <Zap className="w-3 h-3 text-white" />
              </div>
            </div>
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 via-purple-600 to-indigo-600 bg-clip-text text-transparent">
              COVID-19 Fake News Detector
            </h1>
          </div>
          <p className="text-lg text-gray-600 max-w-2xl mx-auto leading-relaxed">
            Advanced AI-powered analysis using ensemble machine learning to detect misinformation and verify claims about COVID-19
          </p>
        </div>

        {/* API Status Card */}
        <div className="bg-white rounded-2xl shadow-lg border border-gray-100 p-6 mb-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className={`flex items-center gap-3 px-4 py-2 rounded-full transition-all duration-300 ${
                apiStatus === 'connected' 
                  ? 'bg-green-50 text-green-700 border border-green-200' 
                  : apiStatus === 'error'
                  ? 'bg-red-50 text-red-700 border border-red-200'
                  : 'bg-yellow-50 text-yellow-700 border border-yellow-200'
              }`}>
                <div className={`w-3 h-3 rounded-full animate-pulse ${
                  apiStatus === 'connected' 
                    ? 'bg-green-500' 
                    : apiStatus === 'error'
                    ? 'bg-red-500'
                    : 'bg-yellow-500'
                }`}></div>
                <span className="font-medium">
                  {apiStatus === 'connected' && 'AI Model Online & Ready'}
                  {apiStatus === 'error' && 'AI Model Offline'}
                  {apiStatus === 'checking' && 'Checking AI Model...'}
                </span>
              </div>
              
              {apiStatus === 'connected' && (
                <div className="flex items-center gap-6 text-sm text-gray-600">
                  <div className="flex items-center gap-1">
                    <Activity className="w-4 h-4 text-blue-500" />
                    <span>99.5% Accuracy</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Shield className="w-4 h-4 text-green-500" />
                    <span>15,000 Features</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Brain className="w-4 h-4 text-purple-500" />
                    <span>4 AI Models</span>
                  </div>
                </div>
              )}
            </div>
            
            <button
              onClick={checkApiStatus}
              className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-colors"
              title="Refresh Connection"
            >
              <RefreshCw size={18} />
            </button>
          </div>
        </div>

        {/* Main Analysis Card */}
        <div className="bg-white rounded-2xl shadow-xl border border-gray-100 overflow-hidden">
          {/* Input Section */}
          <div className="p-8 border-b border-gray-100">
            <label htmlFor="newsText" className="block text-lg font-semibold text-gray-800 mb-4">
              Enter News Article or Claim
            </label>
            <div className="relative">
              <textarea
                id="newsText"
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Paste your COVID-19 related news article, claim, or statement here for analysis..."
                className="w-full h-40 p-4 border-2 border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-gray-700 placeholder-gray-400 transition-all duration-200"
                disabled={apiStatus !== 'connected'}
              />
              <div className="absolute bottom-3 right-3 text-sm text-gray-400">
                {text.length} characters
              </div>
            </div>
            
            <div className="flex items-center justify-between mt-4">
              <p className="text-sm text-gray-500">
                💡 Tip: Press <kbd className="px-2 py-1 text-xs bg-gray-100 rounded">Ctrl+Enter</kbd> for quick analysis
              </p>
              
              <div className="flex gap-3">
                {text && (
                  <button
                    onClick={clearAll}
                    className="px-4 py-2 text-gray-600 hover:text-gray-800 transition-colors"
                  >
                    Clear
                  </button>
                )}
                <button
                  onClick={analyzeText}
                  disabled={loading || apiStatus !== 'connected' || !text.trim() || text.trim().length < 10}
                  className="inline-flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-xl hover:from-blue-700 hover:to-indigo-700 disabled:from-gray-400 disabled:to-gray-500 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-xl transform hover:-translate-y-0.5 disabled:transform-none"
                >
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Send size={20} />
                      Analyze Text
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="p-6 bg-red-50 border-l-4 border-red-400">
              <div className="flex items-center gap-3">
                <AlertTriangle className="text-red-500" size={24} />
                <div>
                  <h4 className="font-semibold text-red-800">Analysis Error</h4>
                  <p className="text-red-700">{error}</p>
                </div>
              </div>
            </div>
          )}

          {/* Result Display */}
          {result && (
            <div className="p-8 bg-gradient-to-br from-gray-50 to-blue-50">
              <h3 className="text-2xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                <Brain className="text-blue-600" size={28} />
                AI Analysis Result
              </h3>
              
              {/* Main Result Card */}
              <div className={`p-6 rounded-2xl border-2 mb-6 transition-all duration-500 ${
                result.prediction === 'FAKE' 
                  ? 'bg-gradient-to-r from-red-50 to-pink-50 border-red-200' 
                  : 'bg-gradient-to-r from-green-50 to-emerald-50 border-green-200'
              }`}>
                <div className="flex items-center gap-4">
                  {result.prediction === 'FAKE' ? (
                    <div className="p-3 bg-red-100 rounded-full">
                      <AlertTriangle className="text-red-600" size={32} />
                    </div>
                  ) : (
                    <div className="p-3 bg-green-100 rounded-full">
                      <CheckCircle className="text-green-600" size={32} />
                    </div>
                  )}
                  
                  <div className="flex-1">
                    <div className="text-2xl font-bold mb-1">
                      <span className={result.prediction === 'FAKE' ? 'text-red-800' : 'text-green-800'}>
                        {result.prediction === 'FAKE' ? '🚨 POTENTIALLY FAKE' : '✅ LIKELY AUTHENTIC'}
                      </span>
                    </div>
                    <div className="text-lg text-gray-600">
                      {result.prediction === 'FAKE' 
                        ? 'This content may contain misinformation or false claims'
                        : 'This content appears to be factual and reliable'
                      }
                    </div>
                  </div>
                  
                  <div className="text-right">
                    <div className="text-3xl font-bold text-gray-800">
                      {(result.confidence * 100).toFixed(1)}%
                    </div>
                    <div className="text-sm text-gray-500">Confidence</div>
                  </div>
                </div>
              </div>

              {/* Confidence Visualization */}
              <div className="bg-white rounded-xl p-6 shadow-lg">
                <div className="flex justify-between items-center mb-3">
                  <span className="font-semibold text-gray-700">Confidence Level</span>
                  <span className="text-sm font-medium text-gray-600">
                    {getConfidenceLabel(result.confidence)}
                  </span>
                </div>
                
                <div className="relative w-full bg-gray-200 rounded-full h-4 overflow-hidden">
                  <div 
                    className={`h-4 rounded-full bg-gradient-to-r ${getConfidenceColor(result.confidence)} transition-all duration-1000 ease-out relative`}
                    style={{ width: `${result.confidence * 100}%` }}
                  >
                    <div className="absolute inset-0 bg-white opacity-20 animate-pulse"></div>
                  </div>
                </div>
                
                <div className="flex justify-between text-xs text-gray-500 mt-2">
                  <span>Low</span>
                  <span>Medium</span>
                  <span>High</span>
                  <span>Very High</span>
                </div>
              </div>

              {/* Additional Info */}
              <div className="mt-6 p-4 bg-blue-50 rounded-xl border border-blue-200">
                <h4 className="font-semibold text-blue-800 mb-2">ℹ️ How This Works</h4>
                <p className="text-blue-700 text-sm leading-relaxed">
                  Our AI ensemble uses 4 advanced machine learning models (PassiveAggressive, SVM, RandomForest, LogisticRegression) 
                  trained on thousands of verified COVID-19 claims. The analysis considers linguistic patterns, 
                  statistical features, and contextual clues to detect potential misinformation.
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="text-center mt-8 text-gray-500">
          <p className="text-sm">
            Powered by Ultimate Ensemble AI • PassiveAggressive + SVM + RandomForest + Logistic Regression
          </p>
          <p className="text-xs mt-1">
            15,000 Features • Advanced Text Processing • Real-time Analysis
          </p>
        </div>
      </div>
    </div>
  )
}

export default FakeNewsDetector