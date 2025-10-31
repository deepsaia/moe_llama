/**
 * Thread management utilities
 */

export interface Thread {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
}

const STORAGE_KEY = 'moe-llama-threads';

export function loadThreads(): Thread[] {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch {
    return [];
  }
}

export function saveThreads(threads: Thread[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(threads));
  } catch (e) {
    console.error('Failed to save threads:', e);
  }
}

export function createThread(title: string = 'New Chat'): Thread {
  return {
    id: crypto.randomUUID(),
    title,
    createdAt: Date.now(),
    updatedAt: Date.now(),
  };
}

export function updateThread(threads: Thread[], id: string, updates: Partial<Thread>): Thread[] {
  return threads.map(t =>
    t.id === id ? { ...t, ...updates, updatedAt: Date.now() } : t
  );
}

export function deleteThread(threads: Thread[], id: string): Thread[] {
  return threads.filter(t => t.id !== id);
}
