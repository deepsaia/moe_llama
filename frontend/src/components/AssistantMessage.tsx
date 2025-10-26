import { type FC } from 'react';
import { MessagePrimitive } from '@assistant-ui/react';
import { makeMarkdownText } from '@assistant-ui/react-markdown';
import { Bot } from 'lucide-react';
import remarkGfm from 'remark-gfm';

const MarkdownText = makeMarkdownText({
  remarkPlugins: [remarkGfm],
});

export const AssistantMessage: FC = () => {
  return (
    <MessagePrimitive.Root className="mb-6 flex justify-start">
      <div className="flex gap-3 max-w-[80%]">
        <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center">
          <Bot className="w-5 h-5 text-white" />
        </div>
        <div className="flex flex-col flex-1">
          <div className="bg-gray-100 dark:bg-gray-700 text-gray-900 dark:text-gray-100 rounded-2xl rounded-tl-sm px-4 py-2.5 prose prose-sm dark:prose-invert max-w-none">
            <MessagePrimitive.Content components={{ Text: MarkdownText }} />
          </div>
        </div>
      </div>
    </MessagePrimitive.Root>
  );
};
