import React, { useState, useRef, useEffect } from 'react';
import { 
  MessageSquare, 
  FileText, 
  PanelLeftClose, 
  PanelLeftOpen, 
  Sparkles, 
  Send, 
  X,
  Plus,
  Bot
} from 'lucide-react';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";

const AgentIA = () => {
  // All state declarations
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [chats, setChats] = useState([]);
  const [currentChat, setCurrentChat] = useState(null);
  const [showSidebar, setShowSidebar] = useState(true);
  const [attachedFiles, setAttachedFiles] = useState([]);
  
  // Refs
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);
  const textAreaRef = useRef(null);

  // Functions
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() && attachedFiles.length === 0) return;
  
    setLoading(true);
    const userMessage = input;
    setInput('');
    
    // Add user message immediately
    setMessages(prev => [...prev, {
      role: 'user',
      content: userMessage,
      files: attachedFiles
    }]);
    
    try {
      if (attachedFiles.length > 0) {
        const formData = new FormData();
        attachedFiles.forEach(file => formData.append('files', file));
        await fetch('/api/upload', {
          method: 'POST',
          body: formData
        });
      }
  
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          message: userMessage,
          chat_id: currentChat
        })
      });
  
      const data = await response.json();
      
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: data.response
      }]);
  
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
      setAttachedFiles([]);
    }
  };

  const handleNewChat = () => {
    const newChat = {
      id: Date.now(),
      name: `Chat ${chats.length + 1}`,
      messages: []
    };
    setChats(prev => [...prev, newChat]);
    setCurrentChat(newChat.id);
    setMessages([]);
  };

  const handleFileUpload = (event) => {
    const files = Array.from(event.target.files);
    setAttachedFiles(prev => [...prev, ...files]);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex h-screen bg-[#F7F7F8]">
      {/* Sidebar */}
      <div 
        className={`${
          showSidebar ? 'w-80' : 'w-0'
        } bg-white border-r border-gray-100 transition-all duration-300 flex flex-col overflow-hidden`}
      >
        <div className="p-4 border-b flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Avatar className="h-8 w-8">
              <AvatarImage src="/assets/minsait_logo.png" alt="Minsait" />
              <AvatarFallback>MS</AvatarFallback>
            </Avatar>
            <span className="font-medium text-gray-900">Agentic RAG</span>
          </div>
        </div>
        
        <div className="p-4">
          <Button
            className="w-full justify-start text-gray-600 bg-gray-50 hover:bg-gray-100"
            onClick={handleNewChat}
          >
            <Plus className="w-4 h-4 mr-2" />
            Nuevo Chat
          </Button>
        </div>

        <ScrollArea className="flex-1 px-3">
          {chats.map(chat => (
            <button
              key={chat.id}
              onClick={() => setCurrentChat(chat.id)}
              className={`w-full p-3 rounded-lg mb-1 text-left flex items-center space-x-3 transition-colors ${
                currentChat === chat.id 
                  ? 'bg-gray-100 text-gray-900' 
                  : 'text-gray-600 hover:bg-gray-50'
              }`}
            >
              <MessageSquare className="w-4 h-4 flex-shrink-0" />
              <span className="text-sm font-medium truncate">{chat.name}</span>
            </button>
          ))}
        </ScrollArea>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="h-14 bg-white border-b border-gray-100 flex items-center px-4 justify-between">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setShowSidebar(!showSidebar)}
                  className="text-gray-600"
                >
                  {showSidebar ? <PanelLeftClose className="h-5 w-5" /> : <PanelLeftOpen className="h-5 w-5" />}
                </Button>
              </TooltipTrigger>
              <TooltipContent>
                {showSidebar ? 'Ocultar sidebar' : 'Mostrar sidebar'}
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
          
          <div className="absolute left-1/2 transform -translate-x-1/2">
            <h1 className="text-sm font-medium text-gray-600">Agentic Compliance Tool</h1>
          </div>
        </header>

        {/* Messages Area */}
        <ScrollArea className="flex-1 py-4">
          <div className="max-w-3xl mx-auto px-4 space-y-6">
            {messages.map((message, index) => (
              <div key={index} className="flex items-start space-x-4">
                <Avatar className={message.role === 'assistant' ? 'bg-[#5E43FF] text-white' : 'bg-gray-100'}>
                  {message.role === 'assistant' ? (
                    <Sparkles className="h-4 w-4" />
                  ) : (
                    <div className="font-medium text-sm">U</div>
                  )}
                </Avatar>
                <div className="flex-1 space-y-2">
                  <div className="font-medium text-sm text-gray-900">
                    {message.role === 'assistant' ? 'Assistant' : 'You'}
                  </div>
                  <div className="text-gray-700 prose prose-sm">
                    {message.content}
                  </div>
                  {message.files?.length > 0 && (
                    <div className="flex flex-wrap gap-2 mt-2">
                      {message.files.map((file, i) => (
                        <div key={i} className="flex items-center gap-2 text-sm bg-gray-50 rounded-lg px-3 py-1">
                          <FileText className="w-4 h-4 text-gray-500" />
                          <span className="text-gray-600">{file.name}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        {/* Input Area */}
        <div className="border-t bg-white p-4">
          <div className="max-w-3xl mx-auto">
            {attachedFiles.length > 0 && (
              <div className="mb-4 space-y-2">
                {attachedFiles.map((file, index) => (
                  <div key={index} className="flex items-center gap-2 text-sm bg-gray-50 rounded-lg px-3 py-2">
                    <FileText className="w-4 h-4 text-gray-500" />
                    <span className="text-gray-600 flex-1 truncate">{file.name}</span>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 hover:bg-gray-200"
                      onClick={() => setAttachedFiles(prev => prev.filter((_, i) => i !== index))}
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </div>
                ))}
              </div>
            )}
            <div className="relative flex items-end gap-2">
              <Dialog>
                <DialogTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="text-gray-500 hover:text-gray-600"
                  >
                    <FileText className="h-5 w-5" />
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Subir archivos</DialogTitle>
                  </DialogHeader>
                  <div className="grid gap-4 py-4">
                    <div className="flex items-center gap-4">
                      <input
                        type="file"
                        ref={fileInputRef}
                        onChange={handleFileUpload}
                        multiple
                        className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                      />
                    </div>
                  </div>
                </DialogContent>
              </Dialog>
              
              <div className="flex-1 overflow-hidden rounded-lg border border-gray-200 bg-white focus-within:border-[#5E43FF] focus-within:ring-1 focus-within:ring-[#5E43FF]">
                <textarea
                  ref={textAreaRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={handleKeyPress}
                  placeholder="Escribe tu mensaje..."
                  className="max-h-48 w-full resize-none border-0 bg-transparent px-4 py-3 text-gray-900 placeholder:text-gray-400 focus:outline-none sm:text-sm"
                  style={{ height: '56px' }}
                  rows={1}
                />
              </div>
              
              <Button
                onClick={handleSend}
                disabled={loading}
                className="bg-[#5E43FF] hover:bg-[#4E35CC] text-white h-[56px] px-4"
              >
                {loading ? (
                  <Bot className="h-5 w-5 animate-pulse" />
                ) : (
                  <Send className="h-5 w-5" />
                )}
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgentIA;