"use client";

import { useState, useRef, useEffect } from "react";
import { 
  Send, Bot, User, FileText, Loader2, BrainCircuit, 
  Plus, MessageSquare, ChevronRight, X 
} from "lucide-react";
import ReactMarkdown from "react-markdown";

// --- Types ---
type Message = {
  role: "user" | "assistant" | "system";
  content: string;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [statusMsg, setStatusMsg] = useState(""); 
  const [uploadedFile, setUploadedFile] = useState<string | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Auto-scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, statusMsg]);

  // --- LOGIC: SEND MESSAGE ---
  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMsg: Message = { role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);
    setStatusMsg("");

    try {
      const res = await fetch(`${API_URL}/chat/`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [userMsg],
          model: "gpt-3.5-turbo",
        }),
      });

      if (!res.ok) throw new Error(`API Error: ${res.status}`);
      
      const data = await res.json();
      const intent = data.intent;

      if (intent === "CHAT") {
        setMessages((prev) => [...prev, { role: "assistant", content: data.content }]);
        setIsLoading(false);
      } 
      else if (intent === "RAG") {
        const jobId = data.job_id;
        setStatusMsg("Analyzing documents & generating response...");
        pollJobStatus(jobId);
      }

    } catch (error) {
      console.error(error);
      setMessages((prev) => [...prev, { role: "assistant", content: "❌ Error connecting to Synapse Core." }]);
      setIsLoading(false);
    }
  };

  // --- LOGIC: POLLING ---
  const pollJobStatus = async (jobId: string) => {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_URL}/chat/tasks/${jobId}`);
        if (!res.ok) return;

        const jobData = await res.json();
        
        if (jobData.status === "complete") {
          clearInterval(interval);
          setMessages((prev) => [...prev, { role: "assistant", content: jobData.result }]);
          setIsLoading(false);
          setStatusMsg("");
        } else if (jobData.status === "failed") {
          clearInterval(interval);
          setMessages((prev) => [...prev, { role: "assistant", content: "❌ Agent Workflow Failed." }]);
          setIsLoading(false);
          setStatusMsg("");
        }
      } catch (e) {
        console.error("Polling error", e);
        clearInterval(interval);
        setIsLoading(false);
      }
    }, 2000); 
  };

  // --- LOGIC: FILE UPLOAD ---
  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    
    const file = e.target.files[0];
    if (file.type !== "application/pdf") {
      alert("Only PDF files are supported.");
      return;
    }

    setIsLoading(true);
    setStatusMsg(`Uploading ${file.name}...`);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_URL}/documents/ingest`, {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        setUploadedFile(data.filename);
        setMessages(prev => [...prev, { 
          role: "system", 
          content: `✅ **Document Uploaded:** ${data.filename} (${data.pages} pages). Indexed in Vector Store.` 
        }]);
      } else {
        const err = await res.json();
        alert(`Upload failed: ${err.detail || res.statusText}`);
      }
    } catch (error) {
      console.error("Upload error:", error);
      alert("Upload failed. Check console.");
    } finally {
      setIsLoading(false);
      setStatusMsg("");
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  return (
    <div className="flex h-screen bg-[#0a0a0a] text-gray-100 font-sans selection:bg-blue-500/30">
      
      {/* --- LEFT SIDEBAR (Glassmorphism) --- */}
      <aside className="w-80 bg-[#0f0f0f] border-r border-white/5 flex flex-col hidden md:flex">
        {/* Logo Area */}
        <div className="p-6 border-b border-white/5">
          <div className="flex items-center gap-3 text-white">
            <div className="p-2 bg-gradient-to-tr from-blue-600 to-indigo-600 rounded-lg shadow-lg shadow-blue-900/20">
              <BrainCircuit className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="font-bold tracking-tight text-lg">Synapse</h1>
              <div className="flex items-center gap-1.5 mt-0.5">
                <span className="relative flex h-1.5 w-1.5">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-green-500"></span>
                </span>
                <span className="text-[10px] uppercase tracking-wider text-gray-500 font-medium">Enterprise Online</span>
              </div>
            </div>
          </div>
        </div>

        {/* Navigation / History Placeholder */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2">
          <div className="text-xs font-medium text-gray-500 uppercase tracking-wider px-2 mb-2">Active Context</div>
          
          {uploadedFile ? (
             <div className="flex items-center gap-3 p-3 bg-blue-950/20 border border-blue-900/30 rounded-xl text-sm text-blue-200">
               <FileText className="w-4 h-4 shrink-0" />
               <span className="truncate">{uploadedFile}</span>
               <button onClick={() => setUploadedFile(null)} className="ml-auto hover:text-white"><X className="w-3 h-3"/></button>
             </div>
          ) : (
            <div className="p-4 border border-dashed border-white/10 rounded-xl text-center">
              <p className="text-xs text-gray-500 mb-2">No documents active</p>
              <button 
                onClick={() => fileInputRef.current?.click()}
                className="text-xs bg-white/5 hover:bg-white/10 text-white px-3 py-1.5 rounded-md transition-colors inline-flex items-center gap-1"
              >
                <Plus className="w-3 h-3" /> Add PDF
              </button>
            </div>
          )}

          <div className="mt-8 text-xs font-medium text-gray-500 uppercase tracking-wider px-2 mb-2">Recent Sessions</div>
          <button className="w-full flex items-center gap-3 p-2 text-gray-400 hover:text-white hover:bg-white/5 rounded-lg transition-all text-sm group">
            <MessageSquare className="w-4 h-4 opacity-50 group-hover:opacity-100" />
            <span className="truncate">Medical Guidelines 2024</span>
            <ChevronRight className="w-3 h-3 ml-auto opacity-0 group-hover:opacity-50" />
          </button>
        </div>

        {/* Footer User Profile */}
        <div className="p-4 border-t border-white/5 bg-[#0a0a0a]/50">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-gray-700 to-gray-900 border border-white/10 flex items-center justify-center">
              <User className="w-4 h-4 text-gray-300" />
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-white truncate">Lead Engineer</p>
              <p className="text-xs text-gray-500 truncate">admin@synapse.com</p>
            </div>
          </div>
        </div>
      </aside>

      {/* --- MAIN CHAT AREA --- */}
      <main className="flex-1 flex flex-col relative bg-[#0a0a0a]">
        
        {/* Top Gradient Fade */}
        <div className="absolute top-0 left-0 w-full h-32 bg-gradient-to-b from-[#0a0a0a] to-transparent z-10 pointer-events-none" />

        {/* Chat Scroll Area */}
        <div className="flex-1 overflow-y-auto p-4 md:p-8 scroll-smooth">
          <div className="max-w-3xl mx-auto space-y-8 pt-10 pb-4">
            
            {messages.length === 0 && (
              <div className="flex flex-col items-center justify-center h-[50vh] text-center space-y-6 opacity-0 animate-in fade-in slide-in-from-bottom-4 duration-700 fill-mode-forwards">
                <div className="w-20 h-20 bg-gradient-to-tr from-blue-500/10 to-indigo-500/10 rounded-3xl flex items-center justify-center border border-white/5 shadow-2xl shadow-blue-900/20">
                  <BrainCircuit className="w-10 h-10 text-blue-500" />
                </div>
                <div>
                  <h2 className="text-2xl font-bold text-white mb-2">How can I help you today?</h2>
                  <p className="text-gray-500 text-sm max-w-md mx-auto leading-relaxed">
                    I'm Synapse, your advanced medical AI. I can analyze documents, summarize protocols, and answer complex queries using RAG.
                  </p>
                </div>
                <div className="flex gap-2">
                  {["Summarize this PDF", "Find contraindications", "Extract patient data"].map(q => (
                    <button key={q} onClick={() => setInput(q)} className="text-xs bg-white/5 border border-white/5 hover:bg-white/10 px-4 py-2 rounded-full transition-colors text-gray-400 hover:text-white">
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((msg, idx) => (
              <div
                key={idx}
                className={`group flex gap-4 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                {/* Assistant Avatar */}
                {msg.role !== "user" && (
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 shadow-lg ${
                    msg.role === "system" ? "bg-green-900/20 text-green-400 ring-1 ring-green-900/50" : "bg-blue-600/10 text-blue-400 ring-1 ring-blue-900/50"
                  }`}>
                    {msg.role === "system" ? <FileText className="w-4 h-4"/> : <Bot className="w-5 h-5" />}
                  </div>
                )}

                {/* Message Bubble */}
                <div
                  className={`relative px-6 py-4 rounded-2xl max-w-[85%] text-sm leading-7 shadow-xl ${
                    msg.role === "user"
                      ? "bg-blue-600 text-white rounded-br-sm"
                      : "bg-[#111111] border border-white/5 text-gray-200 rounded-bl-sm"
                  }`}
                >
                  <div className="prose prose-invert max-w-none prose-p:leading-7 prose-pre:bg-black/50 prose-pre:border prose-pre:border-white/10 prose-code:text-blue-300 prose-code:bg-blue-900/20 prose-code:px-1 prose-code:rounded prose-strong:text-white">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                </div>

                {/* User Avatar */}
                {msg.role === "user" && (
                  <div className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center shrink-0">
                    <User className="w-4 h-4 text-white" />
                  </div>
                )}
              </div>
            ))}

            {/* Status / Loading Indicator */}
            {isLoading && (
              <div className="flex items-center gap-3 pl-12 animate-in fade-in duration-300">
                <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
                <span className="text-sm text-gray-500">{statusMsg || "Processing..."}</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area (Floating Glass) */}
        <div className="p-4 md:p-6 bg-gradient-to-t from-[#0a0a0a] via-[#0a0a0a] to-transparent">
          <div className="max-w-3xl mx-auto bg-[#141414]/80 backdrop-blur-xl border border-white/10 rounded-2xl p-2 flex items-end gap-2 shadow-2xl shadow-black/50 ring-1 ring-white/5">
            
            {/* Hidden Input, necessary for Sidebar "Add PDF" button */}
            <input 
              type="file" 
              ref={fileInputRef} 
              onChange={handleFileSelect} 
              className="hidden" 
              accept="application/pdf"
            />
            
            {/* Paperclip Button REMOVED here */}

            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              placeholder="Message Synapse..."
              className="flex-1 bg-transparent border-none text-gray-100 text-sm px-4 py-3 focus:outline-none focus:ring-0 placeholder:text-gray-600 resize-none max-h-32 min-h-[44px]"
              rows={1}
              disabled={isLoading}
            />
            
            <button
              onClick={handleSend}
              disabled={isLoading || !input.trim()}
              className={`p-3 rounded-xl transition-all duration-300 ${
                input.trim() 
                  ? "bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-500/20" 
                  : "bg-white/5 text-gray-500 cursor-not-allowed"
              }`}
            >
              {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
            </button>
          </div>
          <p className="text-center text-[10px] text-gray-600 mt-3 font-medium tracking-wide">
            SYNAPSE v1.0 • AI can make mistakes. Please verify important medical info.
          </p>
        </div>
      </main>
    </div>
  );
}