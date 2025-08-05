const StatCard = ({ 
  title, 
  value, 
  subtitle, 
  icon: Icon, 
  color = "blue",
  trend = null,
  className = ""
}) => {
  const colorClasses = {
    blue: {
      bg: "bg-blue-50",
      border: "border-blue-200", 
      text: "text-blue-600",
      iconBg: "bg-blue-100"
    },
    green: {
      bg: "bg-green-50",
      border: "border-green-200",
      text: "text-green-600", 
      iconBg: "bg-green-100"
    },
    purple: {
      bg: "bg-purple-50",
      border: "border-purple-200",
      text: "text-purple-600",
      iconBg: "bg-purple-100"
    },
    yellow: {
      bg: "bg-yellow-50", 
      border: "border-yellow-200",
      text: "text-yellow-600",
      iconBg: "bg-yellow-100"
    },
    red: {
      bg: "bg-red-50",
      border: "border-red-200", 
      text: "text-red-600",
      iconBg: "bg-red-100"
    }
  }

  const classes = colorClasses[color] || colorClasses.blue

  return (
    <div className={`${classes.bg} ${classes.border} border rounded-lg p-4 transition-all duration-200 hover:shadow-md ${className}`}>
      <div className="flex items-center justify-between">
        <div className="flex-1">
          <div className="flex items-center space-x-3">
            {Icon && (
              <div className={`${classes.iconBg} p-2 rounded-lg`}>
                <Icon className={`h-5 w-5 ${classes.text}`} />
              </div>
            )}
            <div>
              <p className="text-sm font-medium text-gray-600">{title}</p>
              <p className={`text-2xl font-bold ${classes.text}`}>{value}</p>
              {subtitle && (
                <p className="text-xs text-gray-500 mt-1">{subtitle}</p>
              )}
            </div>
          </div>
        </div>
        
        {trend && (
          <div className={`text-right ${trend > 0 ? 'text-green-600' : 'text-red-600'}`}>
            <p className="text-sm font-medium">
              {trend > 0 ? '+' : ''}{trend}%
            </p>
            <p className="text-xs text-gray-500">vs baseline</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default StatCard