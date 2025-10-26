import { type FC, useEffect, useState } from 'react';
import { fetchModels, type Model } from '@/lib/chatApi';
import { Bot, ChevronDown } from 'lucide-react';

interface ModelSelectorProps {
  selectedModel: string | null;
  onModelChange: (modelId: string | null) => void;
}

export const ModelSelector: FC<ModelSelectorProps> = ({ selectedModel, onModelChange }) => {
  const [models, setModels] = useState<Model[]>([]);
  const [isOpen, setIsOpen] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchModels()
      .then(setModels)
      .finally(() => setLoading(false));
  }, []);

  const selectedModelData = models.find(m => m.id === selectedModel);

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between gap-2 px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 transition-colors text-sm"
      >
        <div className="flex items-center gap-2 flex-1 min-w-0">
          <Bot className="w-4 h-4 flex-shrink-0 text-gray-600 dark:text-gray-300" />
          <span className="truncate text-left text-gray-900 dark:text-gray-100">
            {loading ? 'Loading...' : selectedModelData ? selectedModelData.name : 'Select Model'}
          </span>
        </div>
        <ChevronDown className={`w-4 h-4 flex-shrink-0 text-gray-600 dark:text-gray-300 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <>
          <div
            className="fixed inset-0 z-10"
            onClick={() => setIsOpen(false)}
          />
          <div className="absolute z-20 top-full mt-1 w-full bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg shadow-lg max-h-60 overflow-y-auto">
            {models.length === 0 ? (
              <div className="px-3 py-2 text-sm text-gray-500 dark:text-gray-400">
                {loading ? 'Loading models...' : 'No models available'}
              </div>
            ) : (
              <>
                <button
                  onClick={() => {
                    onModelChange(null);
                    setIsOpen(false);
                  }}
                  className="w-full text-left px-3 py-2 hover:bg-gray-50 dark:hover:bg-gray-600 text-sm text-gray-900 dark:text-gray-100"
                >
                  Auto (Latest)
                </button>
                {models.map(model => (
                  <button
                    key={model.id}
                    onClick={() => {
                      onModelChange(model.id);
                      setIsOpen(false);
                    }}
                    className={`w-full text-left px-3 py-2 hover:bg-gray-50 dark:hover:bg-gray-600 text-sm ${
                      selectedModel === model.id ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300' : 'text-gray-900 dark:text-gray-100'
                    }`}
                  >
                    <div className="font-medium truncate">{model.name}</div>
                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      {new Date(model.modified).toLocaleDateString()}
                    </div>
                  </button>
                ))}
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
};
