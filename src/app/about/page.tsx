"use client";

import { useEffect, useState } from "react";
import { Rocket } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import Navigation from "@/components/Navigation";
import Footer from "@/components/Footer";

type Star = {
  top: string;
  left: string;
  animationDelay: string;
};

const AboutPage = () => {
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
    <div className="min-h-screen cosmic-bg flex flex-col">
      <Navigation />
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

      <div className="container mx-auto px-4 pt-32 pb-16 flex-grow">
        <div className="max-w-4xl mx-auto space-y-12">
          <div className="text-center space-y-4 animate-slide-in">
            <h1 className="font-display font-bold text-4xl sm:text-5xl">
              About <span className="gradient-text">The Project</span>
            </h1>
            <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
              Built for NASA Space Apps Challenge 2025
            </p>
          </div>
          <Card className="card-gradient border-primary/20 hover:border-primary/40 transition-all">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Rocket className="w-5 h-5 text-primary" />
                Our Mission
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-muted-foreground leading-relaxed">
              <p>
                Our mission is to make the discovery of exoplanets more
                accessible, efficient, and interactive. By leveraging
                NASA&apos;s open-source datasets and cutting-edge AI/ML
                techniques, we aim to automatically analyze vast amounts of
                astronomical data to identify new exoplanets with high accuracy.
              </p>
              <p>
                We strive to empower both researchers and enthusiasts by
                providing an intuitive web interface where users can explore,
                upload, and analyze exoplanetary data. Our goal is not only to
                accelerate scientific discovery but also to foster curiosity and
                understanding of the universe. Through innovation, transparency,
                and collaboration, we hope to contribute to humanity&apos;s
                quest to uncover the hidden worlds beyond our solar system.
              </p>
            </CardContent>
          </Card>

          <Card className="card-gradient border-primary/20">
            <CardHeader>
              <CardTitle>Technology Stack</CardTitle>
              <CardDescription>
                Built with modern, cutting-edge technologies
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <h4 className="text-sm font-semibold mb-3 text-muted-foreground">
                  Frontend
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {[
                    "Next.js 15",
                    "React 19",
                    "TypeScript",
                    "Tailwind CSS",
                    "shadcn/ui",
                    "Recharts",
                  ].map((tech, idx) => (
                    <div
                      key={idx}
                      className="p-3 rounded-lg bg-primary/10 border border-primary/20 text-center hover:bg-primary/15 hover:border-primary/30 transition-all interactive-glow"
                    >
                      <span className="font-medium text-sm">{tech}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h4 className="text-sm font-semibold mb-3 text-muted-foreground">
                  Backend & ML
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {[
                    "FastAPI",
                    "PyTorch",
                    "TabTransformer",
                    "scikit-learn",
                    "Pandas",
                    "NumPy",
                  ].map((tech, idx) => (
                    <div
                      key={idx}
                      className="p-3 rounded-lg bg-secondary/10 border border-secondary/20 text-center hover:bg-secondary/15 hover:border-secondary/30 transition-all"
                    >
                      <span className="font-medium text-sm">{tech}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h4 className="text-sm font-semibold mb-3 text-muted-foreground">
                  Deployment
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                  {["Vercel", "Railway", "NASA Exoplanet Archive"].map(
                    (tech, idx) => (
                      <div
                        key={idx}
                        className="p-3 rounded-lg bg-accent/10 border border-accent/20 text-center hover:bg-accent/15 hover:border-accent/30 transition-all"
                      >
                        <span className="font-medium text-sm">{tech}</span>
                      </div>
                    )
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="card-gradient border-primary/20">
            <CardHeader>
              <CardTitle>Key Features</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="space-y-4 text-muted-foreground">
                <li className="flex items-start gap-3 p-3 rounded-lg hover:bg-primary/5 transition-colors">
                  <div className="w-2 h-2 rounded-full bg-electric-blue mt-2 flex-shrink-0" />
                  <span>
                    <strong className="text-foreground">
                      TabTransformer Architecture:
                    </strong>{" "}
                    State-of-the-art transformer-based model optimized for
                    tabular astronomical data, combining attention mechanisms
                    with numerical feature processing
                  </span>
                </li>
                <li className="flex items-start gap-3 p-3 rounded-lg hover:bg-primary/5 transition-colors">
                  <div className="w-2 h-2 rounded-full bg-secondary mt-2 flex-shrink-0" />
                  <span>
                    <strong className="text-foreground">
                      Real-time Training Progress:
                    </strong>{" "}
                    Monitor model training with live batch-level updates,
                    metrics visualization, and the ability to safely close the
                    tab while training continues on the server
                  </span>
                </li>
                <li className="flex items-start gap-3 p-3 rounded-lg hover:bg-primary/5 transition-colors">
                  <div className="w-2 h-2 rounded-full bg-accent mt-2 flex-shrink-0" />
                  <span>
                    <strong className="text-foreground">
                      NASA Exoplanet Archive Integration:
                    </strong>{" "}
                    Train on real data from Kepler Objects of Interest, K2
                    Mission Candidates, and TESS Objects of Interest with
                    15,000+ labeled exoplanet observations
                  </span>
                </li>
                <li className="flex items-start gap-3 p-3 rounded-lg hover:bg-primary/5 transition-colors">
                  <div className="w-2 h-2 rounded-full bg-primary mt-2 flex-shrink-0" />
                  <span>
                    <strong className="text-foreground">
                      Timestamped Model Versions:
                    </strong>{" "}
                    Each trained model is saved with a timestamp, allowing you
                    to track different training sessions and view current model
                    performance metrics
                  </span>
                </li>
                <li className="flex items-start gap-3 p-3 rounded-lg hover:bg-primary/5 transition-colors">
                  <div className="w-2 h-2 rounded-full bg-electric-blue mt-2 flex-shrink-0" />
                  <span>
                    <strong className="text-foreground">
                      Interactive Predictions:
                    </strong>{" "}
                    Make predictions via manual parameter input or CSV batch
                    upload, with instant classification and confidence scores
                  </span>
                </li>
              </ul>
            </CardContent>
          </Card>

          <Card className="card-gradient border-primary/20">
            <CardHeader>
              <CardTitle>Data Sources</CardTitle>
              <CardDescription>
                Powered by NASA&apos;s open data
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="p-4 rounded-lg bg-secondary/5 border border-secondary/10 hover:bg-secondary/10 hover:border-secondary/20 transition-all">
                <h4 className="font-semibold mb-2 text-foreground">
                  Kepler Objects of Interest (KOI)
                </h4>
                <p className="text-sm text-muted-foreground">
                  The Kepler mission discovered thousands of exoplanet
                  candidates by observing stellar brightness variations.
                </p>
              </div>
              <div className="p-4 rounded-lg bg-secondary/5 border border-secondary/10 hover:bg-secondary/10 hover:border-secondary/20 transition-all">
                <h4 className="font-semibold mb-2 text-foreground">
                  K2 Planets and Candidates
                </h4>
                <p className="text-sm text-muted-foreground">
                  K2 extended Kepler&apos;s mission, surveying different regions
                  of the sky for transiting exoplanets.
                </p>
              </div>
              <div className="p-4 rounded-lg bg-secondary/5 border border-secondary/10 hover:bg-secondary/10 hover:border-secondary/20 transition-all">
                <h4 className="font-semibold mb-2 text-foreground">
                  TESS Objects of Interest (TOI)
                </h4>
                <p className="text-sm text-muted-foreground">
                  TESS surveys the entire sky, identifying exoplanet candidates
                  around nearby bright stars.
                </p>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
      <Footer />
    </div>
  );
};

export default AboutPage;
