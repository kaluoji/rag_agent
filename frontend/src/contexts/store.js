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
  
  // Enhanced functions for managing query history
  addToHistory: (queryObj) => {
    const current = get().recentQueries;
    
    // Create a query history object with better structure
    const newQueryHistory = {
      id: queryObj.id || Date.now().toString() + Math.random().toString(36).substr(2, 9),
      text: queryObj.text || get().query,
      timestamp: new Date().toISOString(),
      response: queryObj.response || get().response,
      // Add metadata for better categorization
      type: queryObj.type || (queryObj.text && queryObj.text.toLowerCase().includes('gap') ? 'gap_analysis' : 'consultation'),
      hasDocuments: queryObj.hasDocuments || false
    };
    
    // Add to the beginning of the array and keep only the most recent 50
    const updated = [newQueryHistory, ...current.filter(q => q.id !== newQueryHistory.id)].slice(0, 50);
    
    set({ recentQueries: updated });
    
    // Save to localStorage for persistence
    try {
      localStorage.setItem('recentQueries', JSON.stringify(updated));
    } catch (e) {
      console.error('Failed to save queries to localStorage', e);
    }
  },
  
  // Enhanced function to load stored queries
  loadStoredQueries: () => {
    try {
      const stored = localStorage.getItem('recentQueries');
      if (stored) {
        const queries = JSON.parse(stored);
        // Validate the structure of loaded queries
        const validQueries = queries.filter(q => 
          q && 
          q.id && 
          q.text && 
          q.timestamp
        );
        set({ recentQueries: validQueries });
      }
    } catch (e) {
      console.error('Failed to load queries from localStorage', e);
      // Clear corrupted data
      localStorage.removeItem('recentQueries');
      set({ recentQueries: [] });
    }
  },
  
  // New function to get a specific query by ID
  getQueryById: (id) => {
    const queries = get().recentQueries;
    return queries.find(q => q.id === id) || null;
  },
  
  // New function to remove a specific query
  removeQueryFromHistory: (id) => {
    const current = get().recentQueries;
    const updated = current.filter(q => q.id !== id);
    
    set({ recentQueries: updated });
    
    // Update localStorage
    try {
      localStorage.setItem('recentQueries', JSON.stringify(updated));
    } catch (e) {
      console.error('Failed to update localStorage after removing query', e);
    }
  },
  
  // New function to clear all history
  clearAllHistory: () => {
    set({ recentQueries: [] });
    try {
      localStorage.removeItem('recentQueries');
    } catch (e) {
      console.error('Failed to clear localStorage', e);
    }
  },
  
  // New function to search within history
  searchInHistory: (searchTerm) => {
    const queries = get().recentQueries;
    if (!searchTerm.trim()) return queries;
    
    const term = searchTerm.toLowerCase();
    return queries.filter(q => 
      q.text.toLowerCase().includes(term) ||
      (q.response && q.response.toLowerCase().includes(term))
    );
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