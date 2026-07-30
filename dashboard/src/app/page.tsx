"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Brain, Copy, CheckCircle2, Terminal, Code, ImageIcon, ArrowRight } from 'lucide-react';

export default function Home() {
  const [copiedInstall, setCopiedInstall] = useState(false);
  const [copiedDev, setCopiedDev] = useState(false);

  const copyToClipboard = (text: string, setter: (val: boolean) => void) => {
    navigator.clipboard.writeText(text);
    setter(true);
    setTimeout(() => setter(false), 2000);
  };

  return (
    <main className="min-h-screen bg-[var(--color-loopy-bg)] text-[var(--color-loopy-text)]">
      
      {/* Header */}
      <header className="fixed top-0 w-full z-50 header-blur">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
          <a href="/" className="flex items-center gap-2 font-bold text-xl tracking-tight text-[var(--color-loopy-text)]">
            <svg viewBox="0 0 24 24" width="24" height="24" aria-hidden="true">
              <path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z" fill="none" stroke="var(--color-brand-teal)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z" fill="none" stroke="var(--color-brand-pink)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
            <span className="tracking-tight">neuromotor</span>
          </a>
          <div className="flex items-center gap-8">
            <nav className="nav-pill hidden md:flex">
              <a href="#workflow">Workflow</a>
              <a href="#developer">Developer</a>
              <a href="#research">Research</a>
            </nav>
            <a href="https://github.com/vivannpartani-cyber/neuromotor-brain-to-image-translation" target="_blank" rel="noreferrer" className="btn-ghost px-4 py-1.5 text-sm">
              GitHub
            </a>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="pt-40 pb-24 px-6 max-w-5xl mx-auto flex flex-col md:flex-row items-center gap-16">
        <div className="flex-1">
          <motion.h1 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="hero-title text-5xl md:text-6xl font-normal leading-[1.1] mb-6"
          >
            Your brain waves. <br />
            <strong>Rendered as images.</strong>
          </motion.h1>
          <motion.p 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-lg text-[var(--color-loopy-muted)] mb-8 leading-relaxed max-w-lg"
          >
            Neuromotor is a <strong>local-first neural decoder</strong>. It reads your fMRI or EEG signals, projects them into CLIP semantic space, and reconstructs exactly what you were looking at using Stable Diffusion.
          </motion.p>
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="flex items-center gap-4"
          >
            <button 
              onClick={() => copyToClipboard("pip3 install git+https://github.com/vivannpartani-cyber/neuromotor-brain-to-image-translation.git", setCopiedInstall)}
              className="btn-accent px-6 py-3 text-sm flex items-center gap-2 shadow-sm"
            >
              {copiedInstall ? "Copied!" : "Copy install command"}
            </button>
            <a href="#developer" className="text-sm font-medium hover:underline text-[var(--color-brand-teal)]">
              Developer API &rarr;
            </a>
          </motion.div>
        </div>

        {/* Hero Visual (Mock Terminal / Dashboard) */}
        <motion.div 
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="flex-1 w-full"
        >
          <div className="bg-white border border-gray-200 rounded-xl shadow-sm overflow-hidden flex flex-col h-[320px]">
            <div className="bg-gray-50 border-b border-gray-200 px-4 py-2 flex items-center justify-between text-xs text-gray-500 font-medium">
              <div className="flex gap-1.5">
                <div className="w-2.5 h-2.5 rounded-full bg-red-400" />
                <div className="w-2.5 h-2.5 rounded-full bg-amber-400" />
                <div className="w-2.5 h-2.5 rounded-full bg-green-400" />
              </div>
              <span>neuromotor · loop</span>
              <span className="bg-gray-200 px-2 py-0.5 rounded text-gray-600">local · your brain</span>
            </div>
            <div className="p-5 flex-1 flex flex-col justify-center gap-4 text-sm bg-[#fafafa]">
              <div className="flex items-start gap-3">
                <Brain className="text-[var(--color-brand-teal)] shrink-0" size={18} />
                <div>
                  <div className="font-medium">1. Loading custom neural signals</div>
                  <div className="text-xs text-gray-500 font-mono">Loaded signals: (1, 5438)</div>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <Code className="text-[var(--color-brand-pink)] shrink-0" size={18} />
                <div>
                  <div className="font-medium">2. Mapping to CLIP space</div>
                  <div className="text-xs text-gray-500 font-mono">Predicted embedding: (1, 1024)</div>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <ImageIcon className="text-amber-500 shrink-0" size={18} />
                <div>
                  <div className="font-medium">3. Stable Diffusion Denoising</div>
                  <div className="text-xs text-gray-500 font-mono">30/30 steps [00:40]</div>
                </div>
              </div>
              <div className="mt-2 bg-green-50 border border-green-200 rounded-md p-2 text-green-700 text-xs font-medium flex justify-between items-center">
                <span>✓ Saved → /Downloads/generated_0000.png</span>
              </div>
            </div>
          </div>
        </motion.div>
      </section>

      {/* Workflow Timeline */}
      <section id="workflow" className="py-24 bg-white border-y border-gray-100">
        <div className="max-w-4xl mx-auto px-6">
          <p className="text-xs font-bold uppercase tracking-wider text-[var(--color-brand-teal)] mb-2 text-center">The Pipeline</p>
          <h2 className="text-3xl font-normal hero-title mb-16 text-center">Thoughts to pixels.</h2>
          
          <div className="space-y-12 relative before:absolute before:inset-0 before:ml-[28px] md:before:ml-1/2 before:-translate-x-px md:before:mx-auto before:h-full before:w-0.5 before:bg-gradient-to-b before:from-transparent before:via-gray-200 before:to-transparent">
            
            {/* Step 1 */}
            <div className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active">
              <div className="flex items-center justify-center w-14 h-14 rounded-full bg-white border-4 border-[#fee2e2] text-red-500 shadow shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 z-10">
                <Brain size={24} />
              </div>
              <div className="w-[calc(100%-4rem)] md:w-[calc(50%-3rem)] bg-gray-50 border border-gray-100 p-6 rounded-2xl shadow-sm transition-transform hover:-translate-y-1">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-bold text-red-500 tracking-wider">01. INPUT</span>
                </div>
                <h3 className="font-semibold text-lg mb-2">fMRI / EEG Signals</h3>
                <p className="text-gray-500 text-sm leading-relaxed">Loads raw brainwave recordings (e.g., Miyawaki 2008 dataset) as flattened Numpy arrays representing the visual cortex.</p>
              </div>
            </div>

            {/* Step 2 */}
            <div className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active">
              <div className="flex items-center justify-center w-14 h-14 rounded-full bg-white border-4 border-[#e0e7ff] text-[var(--color-brand-teal)] shadow shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 z-10">
                <Code size={24} />
              </div>
              <div className="w-[calc(100%-4rem)] md:w-[calc(50%-3rem)] bg-gray-50 border border-gray-100 p-6 rounded-2xl shadow-sm transition-transform hover:-translate-y-1">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-bold text-[var(--color-brand-teal)] tracking-wider">02. MAPPING</span>
                </div>
                <h3 className="font-semibold text-lg mb-2">Neural MLP Decoder</h3>
                <p className="text-gray-500 text-sm leading-relaxed">Projects the high-dimensional neural activity directly into the 1024-dimensional semantic CLIP latent space.</p>
              </div>
            </div>

            {/* Step 3 */}
            <div className="relative flex items-center justify-between md:justify-normal md:odd:flex-row-reverse group is-active">
              <div className="flex items-center justify-center w-14 h-14 rounded-full bg-white border-4 border-[#dcfce7] text-green-500 shadow shrink-0 md:order-1 md:group-odd:-translate-x-1/2 md:group-even:translate-x-1/2 z-10">
                <ImageIcon size={24} />
              </div>
              <div className="w-[calc(100%-4rem)] md:w-[calc(50%-3rem)] bg-gray-50 border border-gray-100 p-6 rounded-2xl shadow-sm transition-transform hover:-translate-y-1">
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xs font-bold text-green-500 tracking-wider">03. RENDER</span>
                </div>
                <h3 className="font-semibold text-lg mb-2">Stable Diffusion</h3>
                <p className="text-gray-500 text-sm leading-relaxed">Generates the final image by conditioning the SD v1.5 U-Net with the predicted CLIP embeddings. Thoughts become pixels.</p>
              </div>
            </div>

          </div>
        </div>
      </section>

      {/* Research Section */}
      <section id="research" className="py-24 bg-[var(--color-loopy-bg)] border-b border-gray-100">
        <div className="max-w-6xl mx-auto px-6">
          <p className="text-xs font-bold uppercase tracking-wider text-[var(--color-brand-teal)] mb-2 text-center">Results</p>
          <h2 className="text-3xl font-normal hero-title mb-8 text-center">Research & Reproducibility</h2>
          <div className="bg-white border border-gray-200 shadow-sm rounded-xl p-8 mb-12 text-center max-w-3xl mx-auto">
             <p className="text-gray-600 mb-6">Read our proof-of-concept implementation in the interactive Jupyter notebook, covering dataset fetching (Miyawaki 2008), Ridge vs MLP models, and the full Stable Diffusion pipeline.</p>
             <a href="/demo.html" target="_blank" rel="noreferrer" className="btn-pear px-6 py-3 inline-block shadow-sm hover:-translate-y-0.5 transition-transform">Read the Notebook</a>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white border border-gray-200 shadow-sm rounded-xl overflow-hidden p-2">
              <img src="/images/comparison_0000.png" alt="Comparison 0" className="w-full h-auto rounded-lg border border-gray-100" />
            </div>
            <div className="bg-white border border-gray-200 shadow-sm rounded-xl overflow-hidden p-2">
              <img src="/images/comparison_0001.png" alt="Comparison 1" className="w-full h-auto rounded-lg border border-gray-100" />
            </div>
            <div className="bg-white border border-gray-200 shadow-sm rounded-xl overflow-hidden p-2">
              <img src="/images/comparison_0002.png" alt="Comparison 2" className="w-full h-auto rounded-lg border border-gray-100" />
            </div>
            <div className="bg-white border border-gray-200 shadow-sm rounded-xl overflow-hidden p-2">
              <img src="/images/comparison_0003.png" alt="Comparison 3" className="w-full h-auto rounded-lg border border-gray-100" />
            </div>
          </div>
        </div>
      </section>

      {/* Getting Started Section */}
      <section id="start" className="py-24 px-6 max-w-4xl mx-auto">
        <p className="text-xs font-bold uppercase tracking-wider text-[var(--color-brand-teal)] mb-2 text-center">Getting Started</p>
        <h2 className="text-3xl font-normal hero-title mb-12 text-center">Install and Run</h2>

        {/* Step 1: Install */}
        <div className="bg-white border border-gray-200 shadow-sm rounded-xl p-8 mb-8 relative">
          <div className="absolute -left-4 top-8 w-8 h-8 rounded-full bg-[var(--color-brand-teal)] text-white flex items-center justify-center font-bold text-sm shadow-md">1</div>
          <h3 className="text-xl font-medium mb-4">Install Neuromotor</h3>
          <p className="text-sm text-gray-600 mb-6 leading-relaxed">
            Install the CLI directly from GitHub using pip. This provides the `neuromotor` command globally on your machine.
          </p>
          <div className="code-block p-5 text-sm overflow-x-auto whitespace-pre leading-relaxed relative group shadow-inner">
            <span className="text-white">pip3 install git+https://github.com/vivannpartani-cyber/neuromotor-brain-to-image-translation.git</span>
            <button 
              onClick={() => copyToClipboard("pip3 install git+https://github.com/vivannpartani-cyber/neuromotor-brain-to-image-translation.git", setCopiedInstall)}
              className="absolute top-4 right-4 bg-white/10 hover:bg-white/20 p-2 rounded transition-colors opacity-0 group-hover:opacity-100"
            >
              {copiedInstall ? <CheckCircle2 size={16} className="text-green-400" /> : <Copy size={16} className="text-white" />}
            </button>
          </div>
        </div>

        {/* Step 2: Choose Path */}
        <div className="relative">
          <div className="absolute -left-4 top-8 w-8 h-8 rounded-full bg-[var(--color-brand-pink)] text-white flex items-center justify-center font-bold text-sm shadow-md z-10">2</div>
          <div className="bg-gray-50 border border-gray-200 shadow-sm rounded-xl p-8 mb-8 ml-0">
            <h3 className="text-xl font-medium mb-4">Choose your path</h3>
            
            {/* Path A: Demo */}
            <div className="mb-8">
              <h4 className="font-semibold text-gray-800 mb-2">Option A: Interactive Demo</h4>
              <p className="text-sm text-gray-600 mb-4 leading-relaxed">
                Run the automated end-to-end demo. This automatically downloads the Miyawaki dataset and generates sample reconstructions.
              </p>
              <div className="code-block p-4 text-sm overflow-x-auto whitespace-pre leading-relaxed relative group shadow-inner">
                <span className="text-[var(--color-brand-pink-light)]">neuromotor demo</span>
              </div>
            </div>

            {/* Path B: Developer */}
            <div>
              <h4 className="font-semibold text-gray-800 mb-2">Option B: Developer API</h4>
              <p className="text-sm text-gray-600 mb-4 leading-relaxed">
                Bring your own data. The CLI extracts the dimensionality of your <code>.npy</code> vectors and runs them through the pipeline.
              </p>
              <div className="code-block p-4 text-sm overflow-x-auto whitespace-pre leading-relaxed relative group shadow-inner">
                <span className="text-gray-500"># Run decoding on your own BCI data</span>{"\n"}
                <span className="text-white">neuromotor dev-decode --signals </span><span className="text-[var(--color-brand-pink-light)]">"./my_data.npy"</span>{"\n\n"}
                <span className="text-gray-500"># Provide a custom trained mapper</span>{"\n"}
                <span className="text-white">neuromotor dev-decode --signals </span><span className="text-[var(--color-brand-pink-light)]">"./my_data.npy"</span><span className="text-white"> --mapper </span><span className="text-[var(--color-brand-teal-light)]">"./weights.pt"</span>
              </div>
            </div>
            
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-gray-200 py-12 px-6">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4 text-sm text-[var(--color-loopy-muted)]">
          <div className="flex items-center gap-2">
            <svg viewBox="0 0 24 24" width="16" height="16" aria-hidden="true">
              <path d="M6 12a6 6 0 0 1 12 0" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
              <path d="M18 12a6 6 0 0 1-12 0" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round"/>
            </svg>
            <span className="font-semibold text-gray-700">Neuromotor</span>
          </div>
          <p>© 2026 Neuromotor Proof-of-Concept. Built by Vivann Partani.</p>
        </div>
      </footer>
    </main>
  );
}
