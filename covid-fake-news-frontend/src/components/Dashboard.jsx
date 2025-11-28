import { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  LineChart, Line
} from 'recharts';
import { Activity, TrendingUp, BarChart2, PieChart as PieChartIcon, RefreshCw, AlertTriangle, CheckCircle } from 'lucide-react';

const Dashboard = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const API_BASE_URL = import.meta.env.VITE_BACKEND_URL;

  const fetchDashboardStats = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${API_BASE_URL}/dashboard_stats`);
      setStats(response.data);
    } catch (err) {
      console.error("Error fetching dashboard stats:", err);
      setError("Failed to load dashboard data. Please try again later.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardStats();
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-emerald-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-rose-50 border-l-4 border-rose-500 p-4 my-4 rounded-r-lg">
        <div className="flex">
          <div className="flex-shrink-0">
            <Activity className="h-5 w-5 text-rose-400" />
          </div>
          <div className="ml-3">
            <p className="text-sm text-rose-700">{error}</p>
            <button 
              onClick={fetchDashboardStats}
              className="mt-2 text-sm font-medium text-rose-700 hover:text-rose-600 underline"
            >
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!stats) return null;

  // Professional Color Palette
  const COLORS = {
    real: '#10B981', // Emerald 500
    fake: '#EF4444', // Rose 500
    primary: '#0F766E', // Teal 700
    secondary: '#64748B', // Slate 500
    accent: '#F59E0B' // Amber 500
  };

  return (
    <div className="space-y-8 animate-fade-in">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center border-b border-slate-200 pb-6 gap-4">
        <div>
          <h2 className="text-2xl font-bold text-slate-800 flex items-center gap-2">
            <Activity className="w-6 h-6 text-emerald-700" />
            Analytics Dashboard
          </h2>
          <p className="text-slate-500 text-sm mt-1">Real-time monitoring of misinformation trends</p>
        </div>
        <button 
          onClick={fetchDashboardStats}
          className="p-2.5 rounded-lg bg-white border border-slate-200 text-slate-600 hover:bg-slate-50 hover:text-emerald-700 transition-all shadow-sm self-end sm:self-auto"
          title="Refresh Data"
        >
          <RefreshCw className="w-5 h-5" />
        </button>
      </div>

      {/* Top Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6">
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 hover:border-emerald-200 transition-colors">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wider">Total Analyzed</h3>
            <div className="p-2 bg-slate-100 rounded-lg">
              <BarChart2 className="w-5 h-5 text-slate-600" />
            </div>
          </div>
          <p className="text-3xl font-bold text-slate-800">{stats.distribution.total.toLocaleString()}</p>
          <p className="text-xs text-slate-400 mt-2 font-medium">Claims processed to date</p>
        </div>
        
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 hover:border-rose-200 transition-colors">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wider">Fake Detected</h3>
            <div className="p-2 bg-rose-50 rounded-lg">
              <AlertTriangle className="w-5 h-5 text-rose-500" />
            </div>
          </div>
          <p className="text-3xl font-bold text-rose-600">{stats.distribution.fake.toLocaleString()}</p>
          <p className="text-xs text-slate-400 mt-2 font-medium">
            {stats.distribution.total > 0 ? ((stats.distribution.fake / stats.distribution.total) * 100).toFixed(1) : 0}% of total volume
          </p>
        </div>

        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200 hover:border-emerald-200 transition-colors">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-bold text-slate-500 uppercase tracking-wider">Verified Real</h3>
            <div className="p-2 bg-emerald-50 rounded-lg">
              <CheckCircle className="w-5 h-5 text-emerald-500" />
            </div>
          </div>
          <p className="text-3xl font-bold text-emerald-600">{stats.distribution.real.toLocaleString()}</p>
          <p className="text-xs text-slate-400 mt-2 font-medium">
            {stats.distribution.total > 0 ? ((stats.distribution.real / stats.distribution.total) * 100).toFixed(1) : 0}% of total volume
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Distribution Chart */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <h3 className="text-lg font-bold text-slate-800 mb-6 flex items-center gap-2">
            <PieChartIcon className="w-5 h-5 text-slate-400" />
            Veracity Distribution
          </h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={[
                    { name: 'Real News', value: stats.distribution.real },
                    { name: 'Fake News', value: stats.distribution.fake }
                  ]}
                  cx="50%"
                  cy="50%"
                  innerRadius={80}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                  stroke="none"
                >
                  <Cell key="cell-real" fill={COLORS.real} />
                  <Cell key="cell-fake" fill={COLORS.fake} />
                </Pie>
                <Tooltip 
                  contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                  itemStyle={{ color: '#1e293b', fontWeight: 600 }}
                />
                <Legend verticalAlign="bottom" height={36} iconType="circle" />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Timeline Chart */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <h3 className="text-lg font-bold text-slate-800 mb-6 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-slate-400" />
            7-Day Trend Analysis
          </h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={stats.timeline} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" vertical={false} />
                <XAxis 
                  dataKey="date" 
                  axisLine={false} 
                  tickLine={false} 
                  tick={{ fill: '#64748B', fontSize: 12 }} 
                  dy={10}
                />
                <YAxis 
                  axisLine={false} 
                  tickLine={false} 
                  tick={{ fill: '#64748B', fontSize: 12 }} 
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                />
                <Legend verticalAlign="top" height={36} />
                <Line 
                  type="monotone" 
                  dataKey="fake_count" 
                  name="Fake News" 
                  stroke={COLORS.fake} 
                  strokeWidth={3} 
                  dot={{ r: 4, fill: COLORS.fake, strokeWidth: 2, stroke: '#fff' }}
                  activeDot={{ r: 6 }}
                />
                <Line 
                  type="monotone" 
                  dataKey="real_count" 
                  name="Real News" 
                  stroke={COLORS.real} 
                  strokeWidth={3} 
                  dot={{ r: 4, fill: COLORS.real, strokeWidth: 2, stroke: '#fff' }}
                  activeDot={{ r: 6 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Keywords Chart */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <h3 className="text-lg font-bold text-slate-800 mb-6">Top Keywords in Misinformation</h3>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={stats.top_keywords} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" horizontal={true} vertical={false} stroke="#f1f5f9" />
                <XAxis type="number" hide />
                <YAxis 
                  dataKey="keyword" 
                  type="category" 
                  width={100} 
                  tick={{ fill: '#475569', fontSize: 13, fontWeight: 500 }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip 
                  cursor={{ fill: '#f8fafc' }}
                  contentStyle={{ backgroundColor: '#fff', borderRadius: '8px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                />
                <Bar 
                  dataKey="count" 
                  fill={COLORS.primary} 
                  radius={[0, 4, 4, 0]} 
                  barSize={24}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Trending Topics */}
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <h3 className="text-lg font-bold text-slate-800 mb-6">Emerging Narratives</h3>
          <div className="space-y-3">
            {stats.trending_topics && stats.trending_topics.length > 0 ? (
              stats.trending_topics.map((topic, index) => (
                <div key={index} className="flex items-center p-4 bg-slate-50 rounded-lg border border-slate-100 hover:border-emerald-200 hover:bg-white hover:shadow-sm transition-all group">
                  <span className="flex-shrink-0 w-8 h-8 flex items-center justify-center bg-white border border-slate-200 text-slate-600 rounded-full text-sm font-bold mr-4 group-hover:border-emerald-200 group-hover:text-emerald-700 transition-colors">
                    {index + 1}
                  </span>
                  <p className="text-slate-700 font-medium group-hover:text-slate-900">{topic}</p>
                </div>
              ))
            ) : (
              <div className="text-center py-12 bg-slate-50 rounded-lg border border-dashed border-slate-200">
                <p className="text-slate-400 text-sm">No trending topics identified yet.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
