import { type FC, useEffect, useState } from 'react';
import { Plus, MessageSquare, Trash2, Sparkles } from 'lucide-react';
import { threadStore, type Thread } from '@/lib/threadStore';
import { ModelSelector } from './ModelSelector';
import { ThemeToggle } from './ThemeToggle';

interface SidebarProps {
  currentThreadId: string | null;
  onThreadChange: (threadId: string) => void;
  onNewThread: () => void;
  onClearThread: () => void;
  selectedModel: string | null;
  onModelChange: (modelId: string | null) => void;
}

export const Sidebar: FC<SidebarProps> = ({
  currentThreadId,
  onThreadChange,
  onNewThread,
  onClearThread,
  selectedModel,
  onModelChange,
}) => {
  const [threads, setThreads] = useState<Thread[]>([]);

  const refreshThreads = () => {
    setThreads(threadStore.getAllThreads());
  };

  useEffect(() => {
    refreshThreads();

    // Refresh threads every second to catch updates
    const interval = setInterval(refreshThreads, 1000);
    return () => clearInterval(interval);
  }, []);

  const handleNewThread = () => {
    onNewThread();
    refreshThreads();
  };

  const handleDeleteThread = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    threadStore.deleteThread(id);
    refreshThreads();
    if (currentThreadId === id) {
      // Just clear the current thread, don't auto-create a new one
      onClearThread();
    }
  };

  return (
    <div className="w-64 border-r border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 flex flex-col h-full">
      {/* Header */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-semibold text-sm text-gray-900 dark:text-gray-100">MoE LLaMA</h1>
              <p className="text-xs text-gray-500 dark:text-gray-400">Chat Interface</p>
            </div>
          </div>
          <ThemeToggle />
        </div>

        {/* Model Selector */}
        <ModelSelector selectedModel={selectedModel} onModelChange={onModelChange} />
      </div>

      {/* New Chat Button */}
      <div className="p-3 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
        <button
          onClick={handleNewThread}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-lg bg-blue-600 text-white hover:bg-blue-700 dark:bg-blue-500 dark:hover:bg-blue-600 transition-colors text-sm font-medium"
        >
          <Plus className="w-4 h-4" />
          New Chat
        </button>
      </div>

      {/* Threads List */}
      <div className="flex-1 overflow-y-auto p-3">
        <div className="space-y-1">
          {threads.length === 0 ? (
            <div className="text-center py-8 text-sm text-gray-500 dark:text-gray-400">
              No conversations yet
            </div>
          ) : (
            threads.map(thread => (
              <div
                key={thread.id}
                onClick={() => onThreadChange(thread.id)}
                className={`group flex items-center gap-2 px-3 py-2 rounded-lg cursor-pointer transition-colors ${
                  currentThreadId === thread.id
                    ? 'bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                    : 'hover:bg-white dark:hover:bg-gray-800 text-gray-700 dark:text-gray-300'
                }`}
              >
                <MessageSquare className="w-4 h-4 flex-shrink-0" />
                <span className="flex-1 truncate text-sm">{thread.title}</span>
                <button
                  onClick={(e) => handleDeleteThread(thread.id, e)}
                  className="opacity-0 group-hover:opacity-100 p-1 hover:bg-red-50 dark:hover:bg-red-900/30 rounded transition-opacity"
                >
                  <Trash2 className="w-3 h-3 text-red-600 dark:text-red-400" />
                </button>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="p-3 border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
        <div className="text-xs text-gray-500 dark:text-gray-400 text-center">
          Mixture of Experts LLM
        </div>
      </div>
    </div>
  );
};
