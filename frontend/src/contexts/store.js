import { create } from 'zustand'

export const useQueryStore = create((set) => ({
  query: '',
  response: null,
  isLoading: false,
  error: null,
  setQuery: (query) => set({ query }),
  setResponse: (response) => set({ response }),
  startLoading: () => set({ isLoading: true, error: null }),
  stopLoading: () => set({ isLoading: false }),
  setError: (error) => set({ error, isLoading: false }),
  reset: () => set({ query: '', response: null, isLoading: false, error: null }),
}))

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