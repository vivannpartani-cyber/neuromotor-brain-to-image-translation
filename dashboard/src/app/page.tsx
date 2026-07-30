"use client";

import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Terminal, Brain, Image as ImageIcon, Code, ArrowRight, Globe, Copy, CheckCircle2 } from 'lucide-react';

export default function Home() {
  const [copiedInstall, setCopiedInstall] = useState(false);
  const [copiedDev, setCopiedDev] = useState(false);

  const copyToClipboard = (text: string, setter: (val: boolean) => void) => {
    navigator.clipboard.writeText(text);
    setter(true);
    setTimeout(() => setter(false), 2000);
  };

  return (
    <main className="min-h-screen bg-[var(--color-bg-dark)]">
      
      {/* Navigation */}
      <nav className="fixed top-0 w-full z-50 glass-panel border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2 font-bold text-xl tracking-tight">
            <Brain className="text-[var(--color-brand-cyan)]" />
            <span>Neuromotor</span>
          </div>
          <div className="flex gap-6 text-sm font-medium text-gray-300">
            <a href="#research" className="hover:text-white transition-colors">Research</a>
            <a href="#developers" className="hover:text-white transition-colors">Developers</a>
            <a href="https://github.com/vivannpartani-cyber/neuromotor-brain-to-image-translation" target="_blank" rel="noreferrer" className="flex items-center gap-2 hover:text-white transition-colors">
              <Globe size={18} /> GitHub
            </a>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative pt-40 pb-20 px-6 max-w-7xl mx-auto flex flex-col items-center text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
        >
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/10 text-sm text-gray-300 mb-8">
            <span className="w-2 h-2 rounded-full bg-[var(--color-brand-cyan)] animate-pulse" />
            Stanford AIMI Proof-of-Concept
          </div>
          
          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tighter mb-6 leading-tight">
            Decoding the <span className="text-gradient">Visual Cortex</span> <br />
            into Images
          </h1>
          
          <p className="text-lg md:text-xl text-gray-400 max-w-2xl mx-auto mb-12">
            An end-to-end neural pipeline that maps fMRI brain activity directly to Stable Diffusion semantic space. See what the brain sees.
          </p>

          <div className="neon-border rounded-xl p-1 inline-block max-w-full">
            <div className="glass-panel rounded-lg p-6 flex flex-col md:flex-row items-center gap-6 text-left max-w-2xl">
              <div className="flex-1">
                <p className="text-sm text-gray-400 mb-2 font-medium">Install & run the demo (macOS / Linux / WSL):</p>
                <div className="code-block bg-black/50 p-4 rounded-md border border-white/10 text-[var(--color-brand-cyan)] text-sm flex items-center justify-between group overflow-x-auto whitespace-pre">
                  <span>
                    pip3 install git+https://github.com/vivannpartani-cyber/neuromotor-brain-to-image-translation.git<br/>
                    neuromotor demo
                  </span>
                  <button 
                    onClick={() => copyToClipboard("pip3 install git+https://github.com/vivannpartani-cyber/neuromotor-brain-to-image-translation.git\nneuromotor demo", setCopiedInstall)}
                    className="ml-4 p-2 rounded-md hover:bg-white/10 transition-colors shrink-0"
                  >
                    {copiedInstall ? <CheckCircle2 size={16} className="text-green-400" /> : <Copy size={16} className="text-gray-400 group-hover:text-white" />}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      </section>

      {/* Pipeline Visualisation */}
      <section className="py-20 px-6 border-t border-white/5 bg-black/20">
        <div className="max-w-5xl mx-auto">
          <div className="flex flex-col md:flex-row items-center justify-between gap-12">
            <div className="flex flex-col items-center gap-4 flex-1">
              <div className="w-20 h-20 rounded-2xl glass-panel flex items-center justify-center text-[var(--color-brand-purple)]">
                <Brain size={40} />
              </div>
              <h3 className="font-bold text-lg">1. fMRI Voxels</h3>
              <p className="text-sm text-gray-400 text-center">5,438 dimensional vector from Miyawaki 2008 V1-V4</p>
            </div>
            <ArrowRight className="hidden md:block text-gray-600" />
            <div className="flex flex-col items-center gap-4 flex-1">
              <div className="w-20 h-20 rounded-2xl glass-panel flex items-center justify-center text-[var(--color-brand-cyan)] neon-border">
                <Code size={40} />
              </div>
              <h3 className="font-bold text-lg">2. MLP Mapper</h3>
              <p className="text-sm text-gray-400 text-center">Learns non-linear mapping to CLIP embedding space</p>
            </div>
            <ArrowRight className="hidden md:block text-gray-600" />
            <div className="flex flex-col items-center gap-4 flex-1">
              <div className="w-20 h-20 rounded-2xl glass-panel flex items-center justify-center text-[var(--color-brand-blue)]">
                <ImageIcon size={40} />
              </div>
              <h3 className="font-bold text-lg">3. Image Generation</h3>
              <p className="text-sm text-gray-400 text-center">Stable Diffusion v1.5 conditioned on brain features</p>
            </div>
          </div>
        </div>
      </section>

      {/* Developer API */}
      <section id="developers" className="py-32 px-6 max-w-7xl mx-auto">
        <div className="glass-panel rounded-3xl p-8 md:p-12 relative overflow-hidden">
          {/* Decorative glow */}
          <div className="absolute -top-40 -right-40 w-96 h-96 bg-[var(--color-brand-purple)]/20 rounded-full blur-[100px]" />
          
          <div className="relative z-10 max-w-3xl">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Developer API <span className="text-[var(--color-brand-cyan)]">Neuromotor for BCI</span></h2>
            <p className="text-gray-400 mb-8 text-lg">
              Bring your own EEG or fMRI signals. We've built a dedicated CLI tool for BCI developers to map custom neural vectors directly into the Stable Diffusion semantic space.
            </p>
            
            <div className="space-y-6">
              <div className="bg-black/60 rounded-xl p-6 border border-white/10">
                <h4 className="text-white font-semibold mb-3 flex items-center gap-2">
                  <Terminal size={18} className="text-[var(--color-brand-purple)]" />
                  Decode custom signals
                </h4>
                <div className="code-block bg-black/80 p-4 rounded-md text-gray-300 text-sm flex items-center justify-between group">
                  <span>neuromotor dev-decode --signals ./my_eeg_data.npy</span>
                  <button 
                    onClick={() => copyToClipboard("neuromotor dev-decode --signals ./my_eeg_data.npy", setCopiedDev)}
                    className="p-2 rounded-md hover:bg-white/10 transition-colors shrink-0"
                  >
                    {copiedDev ? <CheckCircle2 size={16} className="text-green-400" /> : <Copy size={16} className="text-gray-400 group-hover:text-white" />}
                  </button>
                </div>
                <p className="text-sm text-gray-500 mt-3">
                  Pass a numpy array containing your signal vectors. Automatically scales and projects to the image latent space.
                </p>
              </div>
              
              <div className="bg-black/60 rounded-xl p-6 border border-white/10">
                <h4 className="text-white font-semibold mb-3">Custom Trained Mappers</h4>
                <p className="text-sm text-gray-400 mb-3">
                  If you have trained your own PyTorch MLP to map your specific electrode layout to CLIP space, pass it directly:
                </p>
                <div className="code-block bg-black/80 p-4 rounded-md text-[var(--color-brand-blue)] text-sm">
                  neuromotor dev-decode --signals ./data.npy --mapper ./my_weights.pt
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/10 py-12 text-center text-gray-500 text-sm">
        <p>Built for the Stanford AIMI Application by Vivann Partani.</p>
      </footer>
    </main>
  );
}
