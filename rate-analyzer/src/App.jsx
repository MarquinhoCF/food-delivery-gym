import React, { useState, useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart, ReferenceLine } from 'recharts';
import { Calculator, TrendingUp, Info, RefreshCw, Upload, Plus, Trash2 } from 'lucide-react';

const RateFunctionAnalyzer = () => {
  const [timeWindow, setTimeWindow] = useState(600);
  const [targetOrders, setTargetOrders] = useState(200);
  const [baseRate, setBaseRate] = useState(0.05);
  const [peaks, setPeaks] = useState([
    { center: 210, intensity: 0.25, width: 1000 },
    { center: 390, intensity: 0.25, width: 1000 }
  ]);
  const [codeInput, setCodeInput] = useState('');

  const parseCode = (code) => {
    try {
      // Extrair total_orders
      const ordersMatch = code.match(/total_orders\s*=\s*(\d+)/);
      if (ordersMatch) setTargetOrders(parseInt(ordersMatch[1]));

      // Extrair time_window
      const windowMatch = code.match(/time_window\s*=\s*(\d+)/);
      if (windowMatch) setTimeWindow(parseInt(windowMatch[1]));

      // Extrair função de taxa
      const rateMatch = code.match(/rate_function\s*=\s*lambda\s+t:\s*(.+?)(?:\)|$)/s);
      if (rateMatch) {
        const rateExpr = rateMatch[1];
        
        // Extrair taxa base
        const baseMatch = rateExpr.match(/^([\d.]+)/);
        if (baseMatch) setBaseRate(parseFloat(baseMatch[1]));

        // Encontrar todos os picos (expressões exp)
        const expRegex = /np\.exp\s*\(\s*-\s*\(\s*\(\s*t\s*-\s*([\d.]+)\s*\)\s*\*\*\s*2\s*\)\s*\/\s*([\d.]+)\s*\)/g;
        const foundPeaks = [];
        let match;
        
        while ((match = expRegex.exec(rateExpr)) !== null) {
          foundPeaks.push({
            center: parseFloat(match[1]),
            intensity: 0.25, // Será ajustado abaixo
            width: parseFloat(match[2])
          });
        }

        // Tentar extrair intensidade global
        const intensityMatch = rateExpr.match(/\+\s*([\d.]+)\s*\*/);
        if (intensityMatch && foundPeaks.length > 0) {
          const globalIntensity = parseFloat(intensityMatch[1]);
          foundPeaks.forEach(peak => peak.intensity = globalIntensity);
        }

        if (foundPeaks.length > 0) {
          setPeaks(foundPeaks);
        }
      }

      alert(`✅ Código parseado com sucesso!\n${peaks.length} pico(s) detectado(s)`);
    } catch (error) {
      alert('❌ Erro ao parsear código. Verifique o formato.');
      console.error(error);
    }
  };

  const addPeak = () => {
    setPeaks([...peaks, { center: timeWindow / 2, intensity: 0.2, width: 1000 }]);
  };

  const removePeak = (index) => {
    if (peaks.length > 1) {
      setPeaks(peaks.filter((_, i) => i !== index));
    }
  };

  const updatePeak = (index, field, value) => {
    const newPeaks = [...peaks];
    newPeaks[index][field] = parseFloat(value) || 0;
    setPeaks(newPeaks);
  };

  const resetDefaults = () => {
    setTimeWindow(600);
    setTargetOrders(200);
    setBaseRate(0.05);
    setPeaks([
      { center: 210, intensity: 0.25, width: 1000 },
      { center: 390, intensity: 0.25, width: 1000 }
    ]);
  };

  const analysis = useMemo(() => {
    const numPoints = 1000;
    const dt = timeWindow / numPoints;
    
    const rateFunction = (t) => {
      let rate = baseRate;
      peaks.forEach(peak => {
        rate += peak.intensity * Math.exp(-((t - peak.center)**2) / peak.width);
      });
      return rate;
    };

    let integral = 0;
    const data = [];
    let maxRate = 0;
    
    for (let i = 0; i <= numPoints; i++) {
      const t = (i / numPoints) * timeWindow;
      const rate = rateFunction(t);
      integral += rate * dt;
      maxRate = Math.max(maxRate, rate);
      
      data.push({
        time: Math.round(t),
        rate: parseFloat(rate.toFixed(4)),
        expectedOrders: parseFloat(integral.toFixed(2))
      });
    }

    const expectedOrders = integral;
    const avgRate = expectedOrders / timeWindow;
    const scaleFactor = targetOrders / expectedOrders;
    
    const adjustedRateFunction = (t) => {
      return scaleFactor * rateFunction(t);
    };

    const adjustedData = [];
    let adjustedIntegral = 0;
    let adjustedMaxRate = 0;
    
    for (let i = 0; i <= numPoints; i++) {
      const t = (i / numPoints) * timeWindow;
      const rate = adjustedRateFunction(t);
      adjustedIntegral += rate * dt;
      adjustedMaxRate = Math.max(adjustedMaxRate, rate);
      
      adjustedData.push({
        time: Math.round(t),
        rate: parseFloat(rate.toFixed(4)),
        expectedOrders: parseFloat(adjustedIntegral.toFixed(2))
      });
    }

    return {
      original: {
        data,
        expectedOrders: expectedOrders.toFixed(2),
        maxRate: maxRate.toFixed(4),
        avgRate: avgRate.toFixed(4),
      },
      adjusted: {
        data: adjustedData,
        expectedOrders: adjustedIntegral.toFixed(2),
        maxRate: adjustedMaxRate.toFixed(4),
        avgRate: (adjustedIntegral / timeWindow).toFixed(4),
        scaleFactor: scaleFactor.toFixed(4),
        newBaseRate: (baseRate * scaleFactor).toFixed(4),
        adjustedPeaks: peaks.map(p => ({
          ...p,
          adjustedIntensity: (p.intensity * scaleFactor).toFixed(4)
        }))
      }
    };
  }, [timeWindow, targetOrders, baseRate, peaks]);

  const combinedData = useMemo(() => {
    return analysis.original.data.map((item, idx) => ({
      time: item.time,
      original: item.rate,
      adjusted: analysis.adjusted.data[idx].rate,
    }));
  }, [analysis]);

  const generateCode = () => {
    const adjustedPeaks = analysis.adjusted.adjustedPeaks;
    const expTerms = adjustedPeaks.map((peak, idx) => 
      `        np.exp(-((t - ${peak.center})**2) / ${peak.width})  # Pico ${idx + 1}`
    ).join(' +\n');

    // Verificar se todas as intensidades são iguais
    const firstIntensity = adjustedPeaks[0].adjustedIntensity;
    const allSame = adjustedPeaks.every(p => p.adjustedIntensity === firstIntensity);

    if (allSame) {
      return `PoissonOrderGenerator(
    total_orders=${targetOrders},
    time_window=${timeWindow},
    rate_function=lambda t: ${analysis.adjusted.newBaseRate} + ${firstIntensity} * (
${expTerms}
    )
)`;
    } else {
      return `PoissonOrderGenerator(
    total_orders=${targetOrders},
    time_window=${timeWindow},
    rate_function=lambda t: ${analysis.adjusted.newBaseRate} + (
${adjustedPeaks.map((peak, idx) => 
  `        ${peak.adjustedIntensity} * np.exp(-((t - ${peak.center})**2) / ${peak.width})  # Pico ${idx + 1}`
).join(' +\n')}
    )
)`;
    }
  };

  const InputField = ({ label, value, onChange, step = 1, min = 0 }) => (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-2">
        {label}
      </label>
      <input
        type="number"
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
        step={step}
        min={min}
        className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all text-lg font-semibold"
      />
    </div>
  );

  return (
    <div className="w-full max-w-7xl mx-auto p-6 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-xl shadow-lg">
      <div className="bg-white rounded-lg p-6 mb-6 shadow-md">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Calculator className="text-blue-600" size={32} />
            <div>
              <h2 className="text-2xl font-bold text-gray-800">Analisador de Função de Taxa</h2>
              <p className="text-gray-600 text-sm mt-1">
                Cole seu código ou ajuste manualmente cada pico
              </p>
            </div>
          </div>
          <button
            onClick={resetDefaults}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <RefreshCw size={18} />
            Restaurar
          </button>
        </div>
      </div>

      {/* Importador de Código */}
      <div className="bg-white rounded-lg p-6 mb-6 shadow-md">
        <div className="flex items-center gap-2 mb-3">
          <Upload size={20} className="text-purple-600" />
          <h3 className="text-lg font-semibold text-gray-800">Importar Código</h3>
        </div>
        <textarea
          value={codeInput}
          onChange={(e) => setCodeInput(e.target.value)}
          placeholder="Cole aqui o código do PoissonOrderGenerator..."
          className="w-full h-32 px-4 py-3 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500 font-mono text-sm"
        />
        <button
          onClick={() => parseCode(codeInput)}
          className="mt-3 px-6 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors flex items-center gap-2"
        >
          <Upload size={18} />
          Parsear e Carregar
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
        {/* Parâmetros Globais */}
        <div className="bg-white rounded-lg p-6 shadow-md">
          <h3 className="text-lg font-semibold mb-4 text-gray-800 flex items-center gap-2">
            <TrendingUp size={20} className="text-green-600" />
            Parâmetros Globais
          </h3>
          
          <div className="space-y-4">
            <InputField
              label="Janela de Tempo (min)"
              value={timeWindow}
              onChange={setTimeWindow}
              step={1}
              min={1}
            />

            <InputField
              label="Pedidos Desejados"
              value={targetOrders}
              onChange={setTargetOrders}
              step={1}
              min={1}
            />

            <InputField
              label="Taxa Base (pedidos/min)"
              value={baseRate}
              onChange={setBaseRate}
              step={0.01}
              min={0}
            />
          </div>
        </div>

        {/* Configuração de Picos */}
        <div className="bg-white rounded-lg p-6 shadow-md lg:col-span-2">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-800">
              Picos ({peaks.length})
            </h3>
            <button
              onClick={addPeak}
              className="flex items-center gap-2 px-3 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm"
            >
              <Plus size={16} />
              Adicionar Pico
            </button>
          </div>

          <div className="space-y-4 max-h-96 overflow-y-auto pr-2">
            {peaks.map((peak, index) => (
              <div key={index} className="p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg border-2 border-blue-200">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="font-semibold text-gray-800">Pico {index + 1}</h4>
                  {peaks.length > 1 && (
                    <button
                      onClick={() => removePeak(index)}
                      className="p-1 text-red-600 hover:bg-red-100 rounded transition-colors"
                    >
                      <Trash2 size={18} />
                    </button>
                  )}
                </div>

                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <label className="block text-xs font-medium text-gray-600 mb-1">
                      Centro (min)
                    </label>
                    <input
                      type="number"
                      value={peak.center}
                      onChange={(e) => updatePeak(index, 'center', e.target.value)}
                      step={1}
                      className="w-full px-3 py-2 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 text-sm font-semibold"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-600 mb-1">
                      Intensidade
                    </label>
                    <input
                      type="number"
                      value={peak.intensity}
                      onChange={(e) => updatePeak(index, 'intensity', e.target.value)}
                      step={0.01}
                      className="w-full px-3 py-2 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 text-sm font-semibold"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-medium text-gray-600 mb-1">
                      Largura
                    </label>
                    <input
                      type="number"
                      value={peak.width}
                      onChange={(e) => updatePeak(index, 'width', e.target.value)}
                      step={50}
                      className="w-full px-3 py-2 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 text-sm font-semibold"
                    />
                  </div>
                </div>

                <div className="mt-2 text-xs text-gray-600 font-mono bg-white px-2 py-1 rounded">
                  Ajustado: intensidade = {analysis.adjusted.adjustedPeaks[index]?.adjustedIntensity}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Resultados */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div className="bg-white rounded-lg p-6 shadow-md">
          <div className="p-4 bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg border-2 border-yellow-200">
            <h4 className="font-semibold text-gray-800 mb-3 flex items-center gap-2">
              <Info size={18} className="text-orange-600" />
              Função Original
            </h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between items-center">
                <span className="text-gray-600">Pedidos Esperados:</span>
                <span className="font-bold text-xl text-orange-600">{analysis.original.expectedOrders}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Taxa Máxima:</span>
                <span className="font-mono font-bold">{analysis.original.maxRate}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Taxa Média:</span>
                <span className="font-mono font-bold">{analysis.original.avgRate}</span>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-lg p-6 shadow-md">
          <div className="p-4 bg-gradient-to-r from-green-50 to-emerald-50 rounded-lg border-2 border-green-200">
            <h4 className="font-semibold text-gray-800 mb-3 flex items-center gap-2">
              <TrendingUp size={18} className="text-green-600" />
              Função Ajustada
            </h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between items-center bg-white px-3 py-2 rounded">
                <span className="text-gray-600">Fator de Escala:</span>
                <span className="font-bold text-xl text-green-600">×{analysis.adjusted.scaleFactor}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Nova Taxa Base:</span>
                <span className="font-mono font-bold text-green-700">{analysis.adjusted.newBaseRate}</span>
              </div>
              <div className="flex justify-between items-center border-t-2 border-green-300 pt-2 mt-2">
                <span className="text-gray-600">Pedidos Esperados:</span>
                <span className="font-bold text-xl text-green-600">{analysis.adjusted.expectedOrders}</span>
              </div>
            </div>
          </div>

          <div className={`mt-4 p-3 rounded-lg border-2 text-center ${
            Math.abs(parseFloat(analysis.adjusted.expectedOrders) - targetOrders) < 1
              ? 'bg-green-50 border-green-300'
              : 'bg-red-50 border-red-300'
          }`}>
            <div className="text-2xl mb-1">
              {Math.abs(parseFloat(analysis.adjusted.expectedOrders) - targetOrders) < 1 ? '✅' : '⚠️'}
            </div>
            <div className="text-sm font-semibold">
              {Math.abs(parseFloat(analysis.adjusted.expectedOrders) - targetOrders) < 1
                ? 'Perfeito!'
                : `Diferença: ${(parseFloat(analysis.adjusted.expectedOrders) - targetOrders).toFixed(1)}`
              }
            </div>
          </div>
        </div>
      </div>

      {/* Gráfico de Taxa */}
      <div className="bg-white rounded-lg p-6 shadow-md mb-6">
        <h3 className="text-lg font-semibold mb-4 text-gray-800">Taxa de Chegada ao Longo do Tempo</h3>
        <ResponsiveContainer width="100%" height={350}>
          <LineChart data={combinedData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis 
              dataKey="time" 
              label={{ value: 'Tempo (min)', position: 'insideBottom', offset: -5 }}
            />
            <YAxis 
              label={{ value: 'Taxa (pedidos/min)', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip 
              contentStyle={{ backgroundColor: '#fff', border: '2px solid #ddd' }}
              formatter={(value) => value.toFixed(4)}
            />
            <Legend />
            <ReferenceLine 
              y={targetOrders / timeWindow} 
              stroke="#888" 
              strokeDasharray="5 5"
              label="Taxa Média Necessária"
            />
            <Line 
              type="monotone" 
              dataKey="original" 
              stroke="#ff6b6b" 
              strokeWidth={2.5}
              dot={false}
              name="Taxa Original"
            />
            <Line 
              type="monotone" 
              dataKey="adjusted" 
              stroke="#51cf66" 
              strokeWidth={2.5}
              dot={false}
              name="Taxa Ajustada"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Gráfico de Pedidos Acumulados */}
      <div className="bg-white rounded-lg p-6 shadow-md mb-6">
        <h3 className="text-lg font-semibold mb-4 text-gray-800">Pedidos Acumulados</h3>
        <ResponsiveContainer width="100%" height={300}>
          <AreaChart data={analysis.adjusted.data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
            <XAxis dataKey="time" label={{ value: 'Tempo (min)', position: 'insideBottom', offset: -5 }} />
            <YAxis label={{ value: 'Pedidos', angle: -90, position: 'insideLeft' }} />
            <Tooltip />
            <Legend />
            <ReferenceLine y={targetOrders} stroke="#51cf66" strokeDasharray="5 5" strokeWidth={2} label="Meta" />
            <Area type="monotone" dataKey="expectedOrders" stroke="#339af0" fill="#74c0fc" fillOpacity={0.6} name="Pedidos Esperados" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Código Gerado */}
      <div className="bg-gray-900 rounded-lg p-6 shadow-md">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-white">Código Python Ajustado</h3>
          <button
            onClick={() => {
              const code = generateCode();
              navigator.clipboard.writeText(code);
              alert('✅ Código copiado!');
            }}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm"
          >
            📋 Copiar Código
          </button>
        </div>
        <pre className="text-green-400 text-sm overflow-x-auto leading-relaxed">
{generateCode()}
        </pre>
      </div>
    </div>
  );
};

export default RateFunctionAnalyzer;