"use client";

import Link from "next/link";
import { Button } from "@/components/ui/button";
import { ArrowRight, Database, Sparkles } from "lucide-react";
import Features from "@/components/Features";
import Navigation from "@/components/Navigation";
import { useEffect, useState } from "react";
import Footer from "@/components/Footer";

type Star = {
  top: string;
  left: string;
  animationDelay: string;
};

const HomePage = () => {
  const [stars, setStars] = useState<Star[]>([]);

  useEffect(() => {
    const generatedStars = [...Array(50)].map(() => ({
      top: `${Math.random() * 100}%`,
      left: `${Math.random() * 100}%`,
      animationDelay: `${Math.random() * 3}s`,
    }));
    setStars(generatedStars);
  }, []);
  return (
    <>
      <Navigation />
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        <div className="absolute inset-0 cosmic-glow opacity-50" />

        <div className="absolute inset-0">
          {stars.map((star, i) => (
            <div
              key={i}
              className="absolute w-1 h-1 bg-primary/30 rounded-full animate-pulse-glow"
              style={star}
            />
          ))}
        </div>
        <div className="container mx-auto px-4 py-32 relative z-10">
          <div className="max-w-4xl mx-auto text-center space-y-8 animate-slide-in">
            <h1 className="font-display font-bold text-5xl sm:text-6xl lg:text-7xl leading-tight">
              A World Away: Hunting for Exoplanets with{" "}
              <span className="gradient-text">AI</span>
            </h1>

            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Interactive machine learning platform for classifying exoplanets
              using NASA&apos;s Kepler, K2, and TESS mission data. Train models,
              visualize results, and explore the cosmos.
            </p>

            <div className="flex flex-wrap justify-center gap-8 py-8">
              <div className="text-center">
                <div className="text-3xl font-display font-bold text-primary">
                  5000+
                </div>
                <div className="text-sm text-muted-foreground">
                  Confirmed Exoplanets
                </div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-display font-bold text-accent">
                  3
                </div>
                <div className="text-sm text-muted-foreground">
                  NASA Missions
                </div>
              </div>
              <div className="text-center">
                <div className="text-3xl font-display font-bold text-secondary">
                  AI
                </div>
                <div className="text-sm text-muted-foreground">
                  Powered Analysis
                </div>
              </div>
            </div>

            <div className="flex flex-wrap justify-center gap-4">
              <Link href="/train" passHref>
                <Button
                  size="lg"
                  className="bg-primary hover:bg-primary/90 text-primary-foreground glow-electric group"
                >
                  Start Training
                  <ArrowRight className="ml-2 w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </Button>
              </Link>
              <Link href="/predict" passHref>
                <Button
                  size="lg"
                  variant="outline"
                  className="border-primary/50 hover:bg-primary/10"
                >
                  <Database className="mr-2 w-4 h-4" />
                  Explore Data
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>
      <Features />
      <Footer />
    </>
  );
};

export default HomePage;
