import { Shield, Activity, Brain } from 'lucide-react'

const Header = () => {
  return (
    <header className="bg-white shadow-lg border-b-4 border-blue-500">
      <div className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <Shield className="h-10 w-10 text-blue-600" />
              <Brain className="h-5 w-5 text-green-500 absolute -top-1 -right-1" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-gray-800">
                COVID-19 Fake News Detector
              </h1>
              <p className="text-gray-600 text-sm">
                AI-Powered Misinformation Detection System
              </p>
            </div>
          </div>
          
          <div className="hidden md:flex items-center space-x-6">
            <div className="flex items-center space-x-2">
              <Activity className="h-5 w-5 text-green-500" />
              <span className="text-sm text-gray-600">AI Status: Active</span>
            </div>
            <div className="bg-gradient-to-r from-blue-600 to-indigo-600 text-white px-4 py-2 rounded-lg">
              <span className="font-semibold">99.5% Accuracy</span>
            </div>
          </div>
        </div>
        
        <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
          <div className="bg-blue-50 p-3 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">99.5%</div>
            <div className="text-sm text-gray-600">Overall Accuracy</div>
          </div>
          <div className="bg-green-50 p-3 rounded-lg">
            <div className="text-2xl font-bold text-green-600">3,112</div>
            <div className="text-sm text-gray-600">Training Samples</div>
          </div>
          <div className="bg-purple-50 p-3 rounded-lg">
            <div className="text-2xl font-bold text-purple-600">15,000</div>
            <div className="text-sm text-gray-600">AI Features</div>
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header