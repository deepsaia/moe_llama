import { type FC } from 'react';
import { ThreadPrimitive } from '@assistant-ui/react';
import { UserMessage } from './UserMessage';
import { AssistantMessage } from './AssistantMessage';
import { Composer } from './Composer';
import { SavedMessages } from './SavedMessages';
import { threadStore, type StoredMessage } from '@/lib/threadStore';

interface ThreadProps {
  threadId?: string | null;
}

export const Thread: FC<ThreadProps> = ({ threadId }) => {
  const savedMessages: StoredMessage[] = threadId ? threadStore.getMessages(threadId) : [];

  return (
    <ThreadPrimitive.Root className="flex h-full flex-col bg-white dark:bg-gray-800">
      <ThreadPrimitive.Viewport className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-4 py-8">
          {savedMessages.length === 0 && (
            <ThreadPrimitive.Empty>
              <div className="flex flex-col items-center justify-center h-[60vh] text-center">
                <div className="text-6xl mb-4">💬</div>
                <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-2">
                  Welcome to MoE LLaMA Chat
                </h2>
                <p className="text-gray-600 dark:text-gray-400">
                  Start a conversation with your Mixture of Experts language model
                </p>
              </div>
            </ThreadPrimitive.Empty>
          )}

          {/* Show saved messages from localStorage */}
          <SavedMessages messages={savedMessages} />

          {/* Show new messages from runtime */}
          <ThreadPrimitive.Messages
            components={{
              UserMessage,
              AssistantMessage,
            }}
          />
        </div>
      </ThreadPrimitive.Viewport>

      <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
        <div className="mx-auto max-w-3xl px-4 py-4">
          <Composer />
        </div>
      </div>
    </ThreadPrimitive.Root>
  );
};
