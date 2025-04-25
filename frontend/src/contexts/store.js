import { create } from 'zustand'

export const useQueryStore = create((set, get) => ({
  query: '',
  response: null,
  isLoading: false,
  error: null,
  // Add this new property for storing recent queries
  recentQueries: [],
  
  setQuery: (query) => set({ query }),
  
  // Modify setResponse to also track the query in history
  setResponse: (response) => {
    set({ response });
    // Add the query to history when we get a response
    const query = get().query;
    if (query && query.trim() !== '') {
      get().addToHistory({
        text: query,
        response: response
      });
    }
  },
  
  startLoading: () => set({ isLoading: true, error: null }),
  stopLoading: () => set({ isLoading: false }),
  setError: (error) => set({ error, isLoading: false }),
  reset: () => set({ query: '', response: null, isLoading: false, error: null }),
  
  // Add these new functions for managing query history
  addToHistory: (queryObj) => {
    const current = get().recentQueries;
    
    // Create a query history object
    const newQueryHistory = {
      id: queryObj.id || Date.now().toString(),
      text: queryObj.text || get().query,
      timestamp: new Date().toISOString(),
      response: queryObj.response
    };
    
    // Add to the beginning of the array and keep only the most recent 20
    const updated = [newQueryHistory, ...current.filter(q => q.id !== newQueryHistory.id)].slice(0, 20);
    
    set({ recentQueries: updated });
    
    // Save to localStorage for persistence
    try {
      localStorage.setItem('recentQueries', JSON.stringify(updated));
    } catch (e) {
      console.error('Failed to save queries to localStorage', e);
    }
  },
  
  // Load recent queries from localStorage on initialization
  loadStoredQueries: () => {
    try {
      const stored = localStorage.getItem('recentQueries');
      if (stored) {
        set({ recentQueries: JSON.parse(stored) });
      }
    } catch (e) {
      console.error('Failed to load queries from localStorage', e);
    }
  },
}))

// Your existing report store remains unchanged
export const useReportStore = create((set) => ({
  reportData: null,
  reportHtml: null,
  reportPath: null,
  status: 'idle', // 'idle' | 'generating' | 'ready' | 'error'
  annotations: [],
  selectedText: '',
  
  setReportData: (reportData) => set({ reportData }),
  setReportHtml: (reportHtml) => set({ reportHtml }),
  setReportPath: (reportPath) => set({ reportPath }),
  setStatus: (status) => set({ status }),
  
  addAnnotation: (annotation) => set((state) => ({ 
    annotations: [...state.annotations, annotation] 
  })),
  
  updateAnnotation: (id, updatedAnnotation) => set((state) => ({
    annotations: state.annotations.map((ann) => 
      ann.id === id ? { ...ann, ...updatedAnnotation } : ann
    ),
  })),
  
  deleteAnnotation: (id) => set((state) => ({
    annotations: state.annotations.filter((ann) => ann.id !== id),
  })),
  
  setSelectedText: (selectedText) => set({ selectedText }),
  
  reset: () => set({ 
    reportData: null, 
    reportHtml: null, 
    reportPath: null,
    status: 'idle', 
    annotations: [], 
    selectedText: '' 
  }),
}))