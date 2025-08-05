import { Brain, Shield, Zap, TrendingUp, Github, ExternalLink } from 'lucide-react'

const Footer = () => {
  return (
    <footer className="bg-gray-900 text-white mt-16">
      <div className="container mx-auto px-4 py-12">
        {/* Main Footer Content */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 mb-8">
          {/* About Section */}
          <div>
            <div className="flex items-center space-x-2 mb-4">
              <Shield className="h-6 w-6 text-blue-400" />
              <h3 className="text-lg font-semibold">AI Model</h3>
            </div>
            <ul className="space-y-2 text-gray-300 text-sm">
              <li>Ultimate Ensemble Classifier</li>
              <li>99.5% Training Accuracy</li>
              <li>89.5% Test Accuracy</li>
              <li>15,000 Advanced Features</li>
            </ul>
          </div>

          {/* Technology Stack */}
          <div>
            <div className="flex items-center space-x-2 mb-4">
              <Brain className="h-6 w-6 text-green-400" />
              <h3 className="text-lg font-semibold">Technology</h3>
            </div>
            <ul className="space-y-2 text-gray-300 text-sm">
              <li>PassiveAggressive Classifier</li>
              <li>Random Forest Ensemble</li>
              <li>Support Vector Machine</li>
              <li>TF-IDF Vectorization</li>
            </ul>
          </div>

          {/* Performance Metrics */}
          <div>
            <div className="flex items-center space-x-2 mb-4">
              <TrendingUp className="h-6 w-6 text-purple-400" />
              <h3 className="text-lg font-semibold">Performance</h3>
            </div>
            <ul className="space-y-2 text-gray-300 text-sm">
              <li>F1-Score Fake: 99.55%</li>
              <li>F1-Score Real: 99.55%</li>
              <li>3,112 Training Samples</li>
              <li>SMOTE Balanced Dataset</li>
            </ul>
          </div>

          {/* Features */}
          <div>
            <div className="flex items-center space-x-2 mb-4">
              <Zap className="h-6 w-6 text-yellow-400" />
              <h3 className="text-lg font-semibold">Features</h3>
            </div>
            <ul className="space-y-2 text-gray-300 text-sm">
              <li>Real-time Detection</li>
              <li>Character N-grams</li>
              <li>Word N-grams</li>
              <li>Advanced Preprocessing</li>
            </ul>
          </div>
        </div>

        {/* Model Architecture */}
        <div className="border-t border-gray-700 pt-8 mb-8">
          <h3 className="text-xl font-semibold mb-4 text-center">Model Architecture</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-gray-800 p-4 rounded-lg text-center">
              <div className="text-2xl font-bold text-blue-400 mb-2">Input Layer</div>
              <p className="text-sm text-gray-300">Advanced text preprocessing with contraction expansion, URL removal, and normalization</p>
            </div>
            <div className="bg-gray-800 p-4 rounded-lg text-center">
              <div className="text-2xl font-bold text-green-400 mb-2">Feature Engineering</div>
              <p className="text-sm text-gray-300">Combined character (3-5) and word (1-3) n-grams with 15,000 total features</p>
            </div>
            <div className="bg-gray-800 p-4 rounded-lg text-center">
              <div className="text-2xl font-bold text-purple-400 mb-2">Ensemble Output</div>
              <p className="text-sm text-gray-300">Voting classifier combining PassiveAggressive, SVM, and RandomForest</p>
            </div>
          </div>
        </div>

        {/* Training Process */}
        <div className="border-t border-gray-700 pt-8 mb-8">
          <h3 className="text-xl font-semibold mb-4 text-center">Training Process</h3>
          <div className="flex flex-wrap justify-center items-center space-x-6 space-y-2">
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-blue-400 rounded-full"></div>
              <span className="text-sm">Data Collection (32 datasets)</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-green-400 rounded-full"></div>
              <span className="text-sm">SMOTE Balancing</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-yellow-400 rounded-full"></div>
              <span className="text-sm">Feature Engineering</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-purple-400 rounded-full"></div>
              <span className="text-sm">Ensemble Training</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-3 h-3 bg-red-400 rounded-full"></div>
              <span className="text-sm">Model Validation</span>
            </div>
          </div>
        </div>

        {/* Technical Details */}
        <div className="border-t border-gray-700 pt-8 mb-8">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div>
              <h4 className="font-semibold mb-3 text-blue-400">Backend Technology</h4>
              <ul className="text-sm text-gray-300 space-y-1">
                <li>• Flask API with CORS support</li>
                <li>• Scikit-learn machine learning</li>
                <li>• Joblib model serialization</li>
                <li>• Real-time prediction endpoints</li>
              </ul>
            </div>
            <div>
              <h4 className="font-semibold mb-3 text-green-400">Frontend Technology</h4>
              <ul className="text-sm text-gray-300 space-y-1">
                <li>• React 18 with Vite</li>
                <li>• Tailwind CSS styling</li>
                <li>• Lucide React icons</li>
                <li>• Responsive design</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="border-t border-gray-700 pt-8 flex flex-col md:flex-row justify-between items-center">
          <div className="flex items-center space-x-4 mb-4 md:mb-0">
            <div className="flex items-center space-x-2">
              <Brain className="h-5 w-5 text-blue-400" />
              <span className="text-sm">COVID-19 Fake News Detection System</span>
            </div>
            <div className="text-sm text-gray-400">
              Built with AI for public health protection
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="text-sm text-gray-400">
              Model Version: Ultimate v1.0
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-sm text-green-400">Live</span>
            </div>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="border-t border-gray-700 pt-6 mt-6">
          <div className="bg-yellow-900 bg-opacity-50 border border-yellow-600 rounded-lg p-4">
            <h4 className="font-semibold text-yellow-400 mb-2">⚠️ Important Disclaimer</h4>
            <p className="text-sm text-yellow-200">
              This AI system is designed to assist in identifying potential misinformation but should not be the sole basis for determining truth. 
              Always verify important health information with official medical sources and healthcare professionals. 
              The model achieves 89.5% accuracy on test data, meaning approximately 1 in 10 predictions may be incorrect.
            </p>
          </div>
        </div>

        {/* Copyright */}
        <div className="text-center mt-8 text-gray-400 text-sm">
          © 2024 COVID-19 Fake News Detection System. Built for educational and public health purposes.
        </div>
      </div>
    </footer>
  )
}

export default Footer