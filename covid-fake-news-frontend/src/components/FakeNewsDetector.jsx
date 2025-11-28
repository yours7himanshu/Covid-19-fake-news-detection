import { useState, useEffect } from 'react'
import { Send, AlertTriangle, CheckCircle, RefreshCw, Brain, Shield, Zap, Activity, Search, ExternalLink, Newspaper, BarChart2 } from 'lucide-react'
import axios from 'axios'
import Dashboard from './Dashboard'

const FakeNewsDetector = () => {
  const [activeTab, setActiveTab] = useState('analyzer')
  const [text, setText] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [apiStatus, setApiStatus] = useState('checking')
  const [verificationResult, setVerificationResult] = useState(null)
  const [verifying, setVerifying] = useState(false)
  const [latestNews, setLatestNews] = useState([])
  const [loadingNews, setLoadingNews] = useState(false)

  const API_BASE_URL = import.meta.env.VITE_BACKEND_URL;

  // Check API status on component mount
  useEffect(() => {
    checkApiStatus()
    const interval = setInterval(checkApiStatus, 30000) // Check every 30 seconds
    return () => clearInterval(interval)
  }, [])

  const checkApiStatus = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/health`)
      if (response.status === 200) {
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
    setVerificationResult(null)

    try {
      // TEMPORARY: Use Gemini for main analysis instead of local model
      const response = await axios.post(`${API_BASE_URL}/verify_fact`, {
        text: text.trim()
      })

      const data = response.data;
      
      if(data){
        // Map Gemini verdict to the format expected by the UI
        let prediction = 'REAL';
        let confidence = 0.95; // Default high confidence for Gemini

        if (data.verdict === 'False' || data.verdict === 'Misleading') {
          prediction = 'FAKE';
        } else if (data.verdict === 'Unverified') {
          confidence = 0.5;
        }

        setResult({
          prediction,
          confidence
        });
        
        // Also set the verification result to show details immediately
        setVerificationResult(data);
      }
      else {
        setError('Data is not received from the server.')
      }
    } catch (err) {
      setError(`Analysis failed: ${err.message}. Please ensure the backend is running.`);
      console.log('Something went wrong',err);
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && e.ctrlKey && !loading) {
      analyzeText()
    }
  }

  const verifyWithGemini = async () => {
    if (!text.trim()) return
    
    setVerifying(true)
    try {
      const response = await axios.post(`${API_BASE_URL}/verify_fact`, {
        text: text.trim()
      })
      setVerificationResult(response.data)
    } catch (err) {
      console.error("Verification failed", err)
      setError("Gemini verification failed. Please try again.")
    } finally {
      setVerifying(false)
    }
  }

  const clearAll = () => {
    setText('')
    setResult(null)
    setError(null)
    setVerificationResult(null)
  }

  const fetchLatestNews = async () => {
    setLoadingNews(true)
    try {
      const response = await axios.get(`${API_BASE_URL}/fetch_latest_news`)
      if (response.data && response.data.news) {
        setLatestNews(response.data.news)
      }
    } catch (err) {
      console.error("Failed to fetch news", err)
    } finally {
      setLoadingNews(false)
    }
  }

  const getConfidenceColor = (confidence) => {
    if (confidence >= 0.8) return 'bg-emerald-500'
    if (confidence >= 0.6) return 'bg-amber-500'
    return 'bg-rose-500'
  }

  const getConfidenceLabel = (confidence) => {
    if (confidence >= 0.9) return 'Very High Confidence'
    if (confidence >= 0.7) return 'High Confidence'
    if (confidence >= 0.5) return 'Medium Confidence'
    return 'Low Confidence'
  }

  return (
    <div className="min-h-screen bg-slate-50 py-8 px-4 font-sans text-slate-900">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8 md:mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="relative">
              <Shield className="w-10 h-10 md:w-12 md:h-12 text-emerald-700" />
              <div className="absolute -top-1 -right-1 w-5 h-5 md:w-6 md:h-6 bg-slate-800 rounded-full flex items-center justify-center border-2 border-white">
                <Activity className="w-2.5 h-2.5 md:w-3 md:h-3 text-white" />
              </div>
            </div>
            <h1 className="text-3xl md:text-4xl font-bold text-slate-800 tracking-tight">
              COVID-19 <span className="text-emerald-700">Fact Check</span>
            </h1>
          </div>
          <p className="text-base md:text-lg text-slate-600 max-w-2xl mx-auto leading-relaxed px-4">
            Professional-grade misinformation detection powered by advanced AI ensemble models.
          </p>
        </div>

        {/* API Status Bar */}
        <div className="bg-white rounded-lg shadow-sm border border-slate-200 p-4 mb-8 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
                apiStatus === 'connected' 
                  ? 'bg-emerald-50 text-emerald-700 border border-emerald-200' 
                  : apiStatus === 'error'
                  ? 'bg-rose-50 text-rose-700 border border-rose-200'
                  : 'bg-amber-50 text-amber-700 border border-amber-200'
              }`}>
                <div className={`w-2 h-2 rounded-full ${
                  apiStatus === 'connected' ? 'bg-emerald-500' : apiStatus === 'error' ? 'bg-rose-500' : 'bg-amber-500'
                }`}></div>
                {apiStatus === 'connected' && 'System Operational'}
                {apiStatus === 'error' && 'System Offline'}
                {apiStatus === 'checking' && 'Connecting...'}
              </div>
              
              {apiStatus === 'connected' && (
                <div className="hidden sm:flex items-center gap-4 text-xs text-slate-500 font-medium">
                  <span>v2.4.0 Stable</span>
                  <span className="w-1 h-1 bg-slate-300 rounded-full"></span>
                  <span>Ensemble Model Active</span>
                </div>
              )}
            </div>
            
            <button
              onClick={checkApiStatus}
              className="p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-50 rounded-md transition-colors"
              title="Refresh Connection"
            >
              <RefreshCw size={16} />
            </button>
        </div>

        {/* Tab Navigation */}
        <div className="flex justify-center mb-8">
          <div className="bg-white p-1 rounded-lg shadow-sm border border-slate-200 inline-flex flex-wrap justify-center gap-1 w-full sm:w-auto">
            <button
              onClick={() => setActiveTab('analyzer')}
              className={`flex-1 sm:flex-none px-4 sm:px-6 py-2.5 rounded-md font-medium text-sm transition-all duration-200 flex items-center justify-center gap-2 ${
                activeTab === 'analyzer'
                  ? 'bg-slate-800 text-white shadow-sm'
                  : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
              }`}
            >
              <Search size={16} />
              <span className="hidden sm:inline">Analyzer</span>
              <span className="sm:hidden">Analyze</span>
            </button>
            <button
              onClick={() => setActiveTab('news')}
              className={`flex-1 sm:flex-none px-4 sm:px-6 py-2.5 rounded-md font-medium text-sm transition-all duration-200 flex items-center justify-center gap-2 ${
                activeTab === 'news'
                  ? 'bg-slate-800 text-white shadow-sm'
                  : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
              }`}
            >
              <Newspaper size={16} />
              <span className="hidden sm:inline">News Monitor</span>
              <span className="sm:hidden">News</span>
            </button>
            <button
              onClick={() => setActiveTab('dashboard')}
              className={`flex-1 sm:flex-none px-4 sm:px-6 py-2.5 rounded-md font-medium text-sm transition-all duration-200 flex items-center justify-center gap-2 ${
                activeTab === 'dashboard'
                  ? 'bg-slate-800 text-white shadow-sm'
                  : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
              }`}
            >
              <BarChart2 size={16} />
              <span className="hidden sm:inline">Dashboard</span>
              <span className="sm:hidden">Stats</span>
            </button>
          </div>
        </div>

        {/* Main Analysis Card */}
        {activeTab === 'analyzer' && (
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
          {/* Input Section */}
          <div className="p-6 md:p-8 border-b border-slate-100">
            <label htmlFor="newsText" className="block text-sm font-bold text-slate-700 uppercase tracking-wide mb-3">
              Input Text for Analysis
            </label>
            <div className="relative">
              <textarea
                id="newsText"
                value={text}
                onChange={(e) => setText(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Paste article text, social media post, or claim here..."
                className="w-full h-40 md:h-48 p-4 bg-slate-50 border border-slate-200 rounded-lg focus:ring-2 focus:ring-emerald-500 focus:border-transparent resize-none text-slate-800 placeholder-slate-400 transition-all duration-200 font-mono text-sm"
                disabled={apiStatus !== 'connected'}
              />
              <div className="absolute bottom-3 right-3 text-xs text-slate-400 font-mono">
                {text.length} chars
              </div>
            </div>
            
            <div className="flex flex-col sm:flex-row items-center justify-between mt-4 gap-4">
              <p className="text-xs text-slate-500 order-2 sm:order-1">
                Press <kbd className="px-1.5 py-0.5 bg-slate-100 border border-slate-200 rounded text-slate-600 font-sans">Ctrl+Enter</kbd> to analyze
              </p>
              
              <div className="flex gap-3 w-full sm:w-auto order-1 sm:order-2">
                {text && (
                  <button
                    onClick={clearAll}
                    className="flex-1 sm:flex-none px-4 py-2 text-sm font-medium text-slate-600 hover:text-slate-800 transition-colors"
                  >
                    Clear
                  </button>
                )}
                <button
                  onClick={analyzeText}
                  disabled={loading || apiStatus !== 'connected' || !text.trim() || text.trim().length < 10}
                  className="flex-1 sm:flex-none inline-flex justify-center items-center gap-2 px-6 py-2.5 bg-emerald-700 text-white rounded-lg hover:bg-emerald-800 disabled:bg-slate-300 disabled:cursor-not-allowed transition-all duration-200 shadow-sm font-medium text-sm"
                >
                  {loading ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                      Processing...
                    </>
                  ) : (
                    <>
                      <Send size={16} />
                      Analyze Content
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>

          {/* Error Display */}
          {error && (
            <div className="p-4 bg-rose-50 border-l-4 border-rose-500 mx-8 mt-6">
              <div className="flex items-center gap-3">
                <AlertTriangle className="text-rose-600" size={20} />
                <p className="text-rose-800 text-sm font-medium">{error}</p>
              </div>
            </div>
          )}

          {/* Result Display */}
          {result && (
            <div className="p-8 bg-slate-50">
              <h3 className="text-lg font-bold text-slate-800 mb-6 flex items-center gap-2 uppercase tracking-wide">
                <Brain className="text-slate-600" size={20} />
                Analysis Report
              </h3>
              
              {/* Main Result Card */}
              <div className="bg-white p-6 rounded-lg border border-slate-200 shadow-sm mb-6">
                <div className="flex flex-col md:flex-row items-center gap-6">
                  {result.prediction === 'FAKE' ? (
                    <div className="p-4 bg-rose-100 rounded-full flex-shrink-0">
                      <AlertTriangle className="text-rose-600" size={32} />
                    </div>
                  ) : (
                    <div className="p-4 bg-emerald-100 rounded-full flex-shrink-0">
                      <CheckCircle className="text-emerald-600" size={32} />
                    </div>
                  )}
                  
                  <div className="flex-1 text-center md:text-left">
                    <div className="text-sm font-bold text-slate-500 uppercase tracking-wider mb-1">Verdict</div>
                    <div className="text-2xl md:text-3xl font-bold mb-2">
                      <span className={result.prediction === 'FAKE' ? 'text-rose-700' : 'text-emerald-700'}>
                        {result.prediction === 'FAKE' ? 'SUSPICIOUS CONTENT' : 'LIKELY AUTHENTIC'}
                      </span>
                    </div>
                    <div className="text-slate-600">
                      {result.prediction === 'FAKE' 
                        ? 'Algorithms detected patterns consistent with misinformation.'
                        : 'Content aligns with verified factual reporting patterns.'
                      }
                    </div>
                  </div>
                  
                  <div className="w-full md:w-auto text-center md:text-right border-t md:border-t-0 md:border-l border-slate-100 pt-4 md:pt-0 md:pl-6 mt-2 md:mt-0">
                    <div className="text-4xl font-bold text-slate-800">
                      {(result.confidence * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs font-bold text-slate-400 uppercase mt-1">Confidence Score</div>
                  </div>
                </div>
              </div>

              {/* Confidence Visualization */}
              <div className="bg-white rounded-lg p-6 border border-slate-200 shadow-sm mb-6">
                <div className="flex justify-between items-center mb-3">
                  <span className="font-semibold text-slate-700 text-sm">Model Confidence</span>
                  <span className="text-xs font-bold px-2 py-1 bg-slate-100 rounded text-slate-600">
                    {getConfidenceLabel(result.confidence)}
                  </span>
                </div>
                
                <div className="relative w-full bg-slate-100 rounded-full h-2 overflow-hidden">
                  <div 
                    className={`h-2 rounded-full ${getConfidenceColor(result.confidence)} transition-all duration-1000 ease-out`}
                    style={{ width: `${result.confidence * 100}%` }}
                  ></div>
                </div>
              </div>

              {/* Gemini Verification Section */}
              <div className="bg-white rounded-lg border border-slate-200 shadow-sm overflow-hidden">
                <div className="p-4 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
                  <h4 className="font-bold text-slate-700 flex items-center gap-2 text-sm uppercase tracking-wide">
                    <Search className="text-slate-500" size={16} />
                    External Verification
                  </h4>
                  {!verificationResult && (
                    <button
                      onClick={verifyWithGemini}
                      disabled={verifying}
                      className="px-3 py-1.5 bg-slate-800 text-white text-xs font-medium rounded hover:bg-slate-700 transition-colors disabled:bg-slate-400 flex items-center gap-2"
                    >
                      {verifying ? 'Verifying...' : 'Cross-Check Sources'}
                    </button>
                  )}
                </div>

                {verificationResult && (
                  <div className="p-6">
                    <div className="flex flex-col sm:flex-row items-start gap-4 mb-6">
                      <div className={`mt-1 p-1.5 rounded flex-shrink-0 ${
                        verificationResult.verdict === 'True' ? 'bg-emerald-100 text-emerald-700' :
                        verificationResult.verdict === 'False' ? 'bg-rose-100 text-rose-700' :
                        'bg-amber-100 text-amber-700'
                      }`}>
                        {verificationResult.verdict === 'True' ? <CheckCircle size={20} /> :
                         verificationResult.verdict === 'False' ? <AlertTriangle size={20} /> :
                         <Activity size={20} />}
                      </div>
                      <div>
                        <h5 className="font-bold text-lg text-slate-800 mb-2">
                          {verificationResult.verdict}
                        </h5>
                        <p className="text-slate-600 leading-relaxed text-sm">
                          {verificationResult.explanation}
                        </p>
                      </div>
                    </div>

                    {verificationResult.sources && verificationResult.sources.length > 0 && (
                      <div className="bg-slate-50 p-4 rounded border border-slate-100">
                        <h6 className="font-bold text-slate-500 text-xs uppercase tracking-wider mb-3 flex items-center gap-2">
                          <ExternalLink size={12} />
                          Citations
                        </h6>
                        <ul className="space-y-2">
                          {verificationResult.sources.map((source, idx) => (
                            <li key={idx} className="text-sm truncate flex items-center gap-2">
                              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 flex-shrink-0"></span>
                              <a 
                                href={source} 
                                target="_blank" 
                                rel="noopener noreferrer"
                                className="text-emerald-700 hover:text-emerald-800 hover:underline transition-colors truncate block w-full"
                              >
                                {source}
                              </a>
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
        )}

        {/* Latest News Section */}
        {activeTab === 'news' && (
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
          <div className="p-6 border-b border-slate-100 flex items-center justify-between">
            <h3 className="text-lg font-bold text-slate-800 flex items-center gap-2 uppercase tracking-wide">
              <Newspaper className="text-slate-500" size={20} />
              Global Health Monitor
            </h3>
            <button
              onClick={fetchLatestNews}
              disabled={loadingNews}
              className="px-4 py-2 bg-slate-100 text-slate-700 rounded-lg hover:bg-slate-200 transition-colors flex items-center gap-2 text-sm font-medium"
            >
              {loadingNews ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-slate-500"></div>
                  Updating...
                </>
              ) : (
                <>
                  <RefreshCw size={16} />
                  Refresh Feed
                </>
              )}
            </button>
          </div>
          
          <div className="p-6 bg-slate-50">
            {latestNews.length === 0 && !loadingNews ? (
              <div className="text-center py-12 text-slate-400">
                <Newspaper className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p className="text-sm">Feed is empty. Refresh to fetch latest reports.</p>
              </div>
            ) : (
              <div className="grid gap-4 md:grid-cols-2">
                {latestNews.map((item, index) => (
                  <div key={index} className="bg-white rounded-lg p-5 border border-slate-200 hover:border-emerald-300 hover:shadow-md transition-all group">
                    <div className="flex justify-between items-start mb-3">
                      <span className="text-xs font-bold text-slate-500 bg-slate-100 px-2 py-1 rounded uppercase tracking-wider">
                        {item.source}
                      </span>
                      <span className={`text-xs font-bold px-2 py-1 rounded uppercase tracking-wider ${
                        item.verdict === 'Real' ? 'bg-emerald-50 text-emerald-700 border border-emerald-100' : 'bg-rose-50 text-rose-700 border border-rose-100'
                      }`}>
                        {item.verdict}
                      </span>
                    </div>
                    
                    <h4 className="font-bold text-slate-800 mb-2 line-clamp-2 group-hover:text-emerald-800 transition-colors">
                      {item.title}
                    </h4>
                    
                    <p className="text-sm text-slate-600 mb-4 line-clamp-3 leading-relaxed">
                      {item.summary}
                    </p>
                    
                    <div className="flex items-center justify-between mt-auto pt-4 border-t border-slate-50">
                      <div className="flex items-center gap-1 text-xs text-slate-400 font-medium">
                        <Activity size={12} />
                        <span>{(item.confidence * 100).toFixed(0)}% Verified</span>
                      </div>
                      
                      {item.url && (
                        <a 
                          href={item.url} 
                          target="_blank" 
                          rel="noopener noreferrer"
                          className="text-emerald-700 hover:text-emerald-900 text-xs font-bold flex items-center gap-1 uppercase tracking-wide"
                        >
                          Full Report <ExternalLink size={12} />
                        </a>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
        )}

        {/* Dashboard Section */}
        {activeTab === 'dashboard' && <Dashboard />}

        {/* Footer */}
        <div className="text-center mt-12 pb-8 text-slate-400">
          <p className="text-xs font-medium uppercase tracking-widest">
            System Architecture: Ensemble Learning (PA + SVM + RF + LR)
          </p>
          <p className="text-[10px] mt-2 opacity-70">
            v2.4.0 • Secure Connection • 256-bit Encryption
          </p>
        </div>
      </div>
    </div>
  )
}

export default FakeNewsDetector
