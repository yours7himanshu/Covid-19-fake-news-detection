import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

const ConfidenceIndicator = ({ confidence, prediction, className = "" }) => {
  const getConfidenceLevel = (confidence) => {
    if (confidence > 1.5) return { level: "Very High", color: "green", percentage: 95 }
    if (confidence > 1.0) return { level: "High", color: "blue", percentage: 85 }
    if (confidence > 0.5) return { level: "Medium", color: "yellow", percentage: 70 }
    return { level: "Low", color: "red", percentage: 50 }
  }

  const { level, color, percentage } = getConfidenceLevel(confidence)
  
  const colorClasses = {
    green: {
      bg: "bg-green-100",
      border: "border-green-300",
      text: "text-green-800",
      bar: "bg-green-500",
      icon: "text-green-600"
    },
    blue: {
      bg: "bg-blue-100", 
      border: "border-blue-300",
      text: "text-blue-800",
      bar: "bg-blue-500",
      icon: "text-blue-600"
    },
    yellow: {
      bg: "bg-yellow-100",
      border: "border-yellow-300", 
      text: "text-yellow-800",
      bar: "bg-yellow-500",
      icon: "text-yellow-600"
    },
    red: {
      bg: "bg-red-100",
      border: "border-red-300",
      text: "text-red-800", 
      bar: "bg-red-500",
      icon: "text-red-600"
    }
  }

  const classes = colorClasses[color]
  
  const getIcon = () => {
    if (confidence > 1.0) return <TrendingUp className={`h-5 w-5 ${classes.icon}`} />
    if (confidence > 0.5) return <Minus className={`h-5 w-5 ${classes.icon}`} />
    return <TrendingDown className={`h-5 w-5 ${classes.icon}`} />
  }

  return (
    <div className={`${classes.bg} ${classes.border} border rounded-lg p-4 ${className}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          {getIcon()}
          <span className={`font-semibold ${classes.text}`}>
            Confidence: {level}
          </span>
        </div>
        <span className={`text-sm ${classes.text}`}>
          {confidence.toFixed(2)}
        </span>
      </div>
      
      {/* Confidence Bar */}
      <div className="w-full bg-gray-200 rounded-full h-2 mb-2">
        <div 
          className={`${classes.bar} h-2 rounded-full transition-all duration-500 ease-out`}
          style={{ width: `${Math.min(percentage, 100)}%` }}
        ></div>
      </div>
      
      <div className="flex justify-between text-xs text-gray-600">
        <span>Low</span>
        <span>Medium</span>
        <span>High</span>
        <span>Very High</span>
      </div>
      
      <div className={`mt-3 text-sm ${classes.text}`}>
        <p>
          The AI is <strong>{level.toLowerCase()}</strong> confident that this content is{' '}
          <strong>{prediction.toLowerCase()}</strong> news.
        </p>
      </div>
    </div>
  )
}

export default ConfidenceIndicator