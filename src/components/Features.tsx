import { Database, Brain, LineChart, Zap } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

const features = [
  {
    icon: Database,
    title: "NASA Mission Data",
    description:
      "Access data from Kepler, K2, and TESS missions with built-in APIs for exoplanet candidates and confirmed planets.",
  },
  {
    icon: Brain,
    title: "Custom ML Models",
    description:
      "Train your own machine learning models with configurable hyperparameters including learning rate, epochs, and batch size.",
  },
  {
    icon: LineChart,
    title: "Real-time Visualization",
    description:
      "Monitor training progress with interactive charts showing loss, accuracy, and confusion matrices in real-time.",
  },
  {
    icon: Zap,
    title: "Instant Predictions",
    description:
      "Upload new data and get instant exoplanet classifications powered by your trained models.",
  },
];

const Features = () => {
  return (
    <section className="py-24 relative">
      <div className="container mx-auto px-4">
        <div className="text-center mb-16">
          <h2 className="font-display font-bold text-4xl sm:text-5xl mb-4">
            Powerful <span className="gradient-text">Features</span>
          </h2>
          <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
            Everything you need to analyze and classify exoplanets with
            cutting-edge AI technology
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {features.map((feature, index) => {
            const Icon = feature.icon;
            return (
              <Card
                key={index}
                className="bg-card/50 backdrop-blur border-border/40 hover:border-primary/50 transition-all hover:glow-electric group"
              >
                <CardHeader>
                  <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                    <Icon className="w-6 h-6 text-primary" />
                  </div>
                  <CardTitle className="font-display">
                    {feature.title}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-muted-foreground">
                    {feature.description}
                  </CardDescription>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>
    </section>
  );
};

export default Features;
