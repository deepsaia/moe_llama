import { type FC } from 'react';
import { User, Bot } from 'lucide-react';
import { type StoredMessage } from '@/lib/threadStore';

interface SavedMessagesProps {
  messages: StoredMessage[];
}

export const SavedMessages: FC<SavedMessagesProps> = ({ messages }) => {
  if (messages.length === 0) return null;

  return (
    <>
      {messages.map((msg, index) => (
        <div key={`${msg.id}-${index}`} className="mb-6">
          {msg.role === 'user' ? (
            <div className="flex justify-end">
              <div className="flex gap-3 max-w-[80%]">
                <div className="flex flex-col items-end flex-1">
                  <div className="bg-blue-600 dark:bg-blue-500 text-white rounded-2xl rounded-tr-sm px-4 py-2.5">
                    <div className="whitespace-pre-wrap">{msg.content}</div>
                  </div>
                </div>
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
                  <User className="w-5 h-5 text-blue-600 dark:text-blue-400" />
                </div>
              </div>
            </div>
          ) : (
            <div className="flex justify-start">
              <div className="flex gap-3 max-w-[80%]">
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
                  <Bot className="w-5 h-5 text-white" />
                </div>
                <div className="flex flex-col flex-1">
                  <div className="bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-gray-100 rounded-2xl rounded-tl-sm px-4 py-2.5">
                    <div className="whitespace-pre-wrap">{msg.content}</div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      ))}
    </>
  );
};
