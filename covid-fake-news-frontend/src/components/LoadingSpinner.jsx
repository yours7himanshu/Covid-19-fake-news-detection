import { Loader } from 'lucide-react'

const LoadingSpinner = ({ message = "Loading...", size = "medium" }) => {
  const sizeClasses = {
    small: "h-4 w-4",
    medium: "h-8 w-8", 
    large: "h-12 w-12"
  }

  return (
    <div className="flex flex-col items-center justify-center p-8">
      <Loader className={`${sizeClasses[size]} animate-spin text-blue-600 mb-4`} />
      <p className="text-gray-600 text-sm">{message}</p>
    </div>
  )
}

export default LoadingSpinner