import { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import { AssistantRuntimeProvider, useLocalRuntime, type ChatModelAdapter } from '@assistant-ui/react';
import { Thread } from './components/Thread';
import { Sidebar } from './components/Sidebar';
import { streamChatCompletion, type ChatMessage } from './lib/chatApi';
import { threadStore } from './lib/threadStore';

function ChatInterface({ threadId, selectedModel }: { threadId: string | null; selectedModel: string | null }) {
  const threadCreatedRef = useRef(false);
  const currentThreadIdRef = useRef(threadId);
  const savedMessagesCount = useRef(0);

  // Update refs when threadId changes
  useEffect(() => {
    currentThreadIdRef.current = threadId;
    if (threadId) {
      savedMessagesCount.current = threadStore.getMessages(threadId).length;
    } else {
      savedMessagesCount.current = 0;
    }
  }, [threadId]);

  // Create custom adapter for FastAPI backend
  const adapter: ChatModelAdapter = useMemo(() => ({
    async *run({ messages, abortSignal }) {
      // Auto-create thread on first message if none exists
      let activeThreadId = currentThreadIdRef.current;
      if (!activeThreadId && !threadCreatedRef.current) {
        const newThread = threadStore.createThread();
        activeThreadId = newThread.id;
        currentThreadIdRef.current = activeThreadId;
        threadCreatedRef.current = true;
      }

      // Get all saved messages for this thread
      const savedMessages = activeThreadId ? threadStore.getMessages(activeThreadId) : [];

      // Convert runtime messages (new messages only, not saved ones)
      const newRuntimeMessages = messages.slice(savedMessagesCount.current);

      // Save new user message if exists
      if (activeThreadId && newRuntimeMessages.length > 0) {
        const lastMessage = newRuntimeMessages[newRuntimeMessages.length - 1];
        if (lastMessage.role === 'user') {
          const content = lastMessage.content
            .filter((part) => part.type === 'text')
            .map((part) => part.type === 'text' ? part.text : '')
            .join('');

          threadStore.addMessage(activeThreadId, {
            role: 'user',
            content,
          });
          savedMessagesCount.current++;
        }
      }

      // Combine saved messages with runtime messages for API call
      const allApiMessages: ChatMessage[] = [
        ...savedMessages.map(msg => ({
          role: msg.role as 'user' | 'assistant',
          content: msg.content,
        })),
        ...newRuntimeMessages
          .filter((msg) => msg.role === 'user' || msg.role === 'assistant')
          .map((msg) => ({
            role: msg.role as 'user' | 'assistant',
            content: msg.content
              .filter((part) => part.type === 'text')
              .map((part) => part.type === 'text' ? part.text : '')
              .join(''),
          }))
      ];

      // Remove duplicates (in case user message was just added)
      const uniqueMessages: ChatMessage[] = [];
      const seen = new Set<string>();
      for (const msg of allApiMessages) {
        const key = `${msg.role}:${msg.content}`;
        if (!seen.has(key)) {
          seen.add(key);
          uniqueMessages.push(msg);
        }
      }

      // Stream from FastAPI backend with selected model
      const stream = streamChatCompletion(uniqueMessages, {
        model: selectedModel || undefined,
        temperature: 0.8,
        max_tokens: 200,
        top_k: 50,
        top_p: 0.95,
      });

      let fullText = '';
      for await (const chunk of stream) {
        if (abortSignal?.aborted) {
          return;
        }

        fullText += chunk;

        yield {
          content: [
            {
              type: 'text' as const,
              text: fullText,
            },
          ],
        };
      }

      // Save assistant message to thread store
      if (activeThreadId && fullText) {
        threadStore.addMessage(activeThreadId, {
          role: 'assistant',
          content: fullText,
        });
        savedMessagesCount.current++;
      }
    },
  }), [selectedModel]);

  const runtime = useLocalRuntime(adapter);

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <Thread threadId={threadId} />
    </AssistantRuntimeProvider>
  );
}

function App() {
  const [currentThreadId, setCurrentThreadId] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const handleNewThread = useCallback(() => {
    const newThread = threadStore.createThread();
    setCurrentThreadId(newThread.id);
  }, []);

  const handleThreadChange = useCallback((threadId: string) => {
    setCurrentThreadId(threadId);
  }, []);

  return (
    <div className="flex h-full">
      <Sidebar
        currentThreadId={currentThreadId}
        onThreadChange={handleThreadChange}
        onNewThread={handleNewThread}
        onClearThread={() => setCurrentThreadId(null)}
        selectedModel={selectedModel}
        onModelChange={setSelectedModel}
      />
      <div className="flex-1 h-full">
        <ChatInterface key={currentThreadId || 'new'} threadId={currentThreadId} selectedModel={selectedModel} />
      </div>
    </div>
  );
}

export default App;
