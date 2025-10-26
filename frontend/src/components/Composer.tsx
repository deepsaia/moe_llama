import { type FC } from 'react';
import { ComposerPrimitive } from '@assistant-ui/react';
import { Send } from 'lucide-react';

export const Composer: FC = () => {
  return (
    <ComposerPrimitive.Root className="flex items-end gap-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 p-2 shadow-sm focus-within:border-blue-500 dark:focus-within:border-blue-400 focus-within:ring-1 focus-within:ring-blue-500 dark:focus-within:ring-blue-400">
      <ComposerPrimitive.Input
        className="flex-1 resize-none bg-transparent px-2 py-1.5 text-sm outline-none placeholder:text-gray-400 dark:placeholder:text-gray-500 text-gray-900 dark:text-gray-100 max-h-40"
        placeholder="Type your message..."
        rows={1}
        autoFocus
      />
      <ComposerPrimitive.Send className="flex h-8 w-8 items-center justify-center rounded-md bg-blue-600 dark:bg-blue-500 text-white transition-colors hover:bg-blue-700 dark:hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed">
        <Send className="h-4 w-4" />
      </ComposerPrimitive.Send>
    </ComposerPrimitive.Root>
  );
};
