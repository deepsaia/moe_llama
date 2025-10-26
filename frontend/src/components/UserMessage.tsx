import { type FC } from 'react';
import { MessagePrimitive } from '@assistant-ui/react';
import { User } from 'lucide-react';

export const UserMessage: FC = () => {
  return (
    <MessagePrimitive.Root className="mb-6 flex justify-end">
      <div className="flex gap-3 max-w-[80%]">
        <div className="flex flex-col items-end flex-1">
          <div className="bg-blue-600 dark:bg-blue-500 text-white rounded-2xl rounded-tr-sm px-4 py-2.5">
            <MessagePrimitive.Content />
          </div>
        </div>
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center">
          <User className="w-5 h-5 text-blue-600 dark:text-blue-400" />
        </div>
      </div>
    </MessagePrimitive.Root>
  );
};
