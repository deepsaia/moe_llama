/**
 * Thread Store - Manages conversation threads and their messages
 */

export interface StoredMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: number;
}

export interface Thread {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: StoredMessage[];
}

const THREADS_KEY = 'moe-llama-threads';

export class ThreadStore {
  private threads: Map<string, Thread>;

  constructor() {
    this.threads = new Map();
    this.loadFromStorage();
  }

  private loadFromStorage() {
    try {
      const stored = localStorage.getItem(THREADS_KEY);
      if (stored) {
        const threadsArray: Thread[] = JSON.parse(stored);
        threadsArray.forEach(thread => {
          this.threads.set(thread.id, thread);
        });
      }
    } catch (e) {
      console.error('Failed to load threads:', e);
    }
  }

  private saveToStorage() {
    try {
      const threadsArray = Array.from(this.threads.values());
      localStorage.setItem(THREADS_KEY, JSON.stringify(threadsArray));
    } catch (e) {
      console.error('Failed to save threads:', e);
    }
  }

  getThread(id: string): Thread | undefined {
    return this.threads.get(id);
  }

  getAllThreads(): Thread[] {
    return Array.from(this.threads.values()).sort(
      (a, b) => b.updatedAt - a.updatedAt
    );
  }

  createThread(): Thread {
    const thread: Thread = {
      id: crypto.randomUUID(),
      title: 'New Chat',
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: [],
    };
    this.threads.set(thread.id, thread);
    this.saveToStorage();
    return thread;
  }

  updateThread(id: string, updates: Partial<Omit<Thread, 'id' | 'messages'>>) {
    const thread = this.threads.get(id);
    if (thread) {
      Object.assign(thread, updates, { updatedAt: Date.now() });
      this.saveToStorage();
    }
  }

  deleteThread(id: string) {
    this.threads.delete(id);
    this.saveToStorage();
  }

  addMessage(threadId: string, message: Omit<StoredMessage, 'id' | 'timestamp'>) {
    const thread = this.threads.get(threadId);
    if (thread) {
      const storedMessage: StoredMessage = {
        ...message,
        id: crypto.randomUUID(),
        timestamp: Date.now(),
      };
      thread.messages.push(storedMessage);
      thread.updatedAt = Date.now();

      // Auto-update title from first user message
      if (thread.messages.length === 1 && message.role === 'user') {
        thread.title = message.content.slice(0, 50) + (message.content.length > 50 ? '...' : '');
      }

      this.saveToStorage();
      return storedMessage;
    }
  }

  getMessages(threadId: string): StoredMessage[] {
    const thread = this.threads.get(threadId);
    return thread?.messages || [];
  }

  clearMessages(threadId: string) {
    const thread = this.threads.get(threadId);
    if (thread) {
      thread.messages = [];
      thread.updatedAt = Date.now();
      this.saveToStorage();
    }
  }
}

// Singleton instance
export const threadStore = new ThreadStore();
