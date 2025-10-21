"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Rocket, Brain, Target, Info, Terminal } from "lucide-react";
import BackendStatus from "./BackendStatus";

const Navigation = () => {
  const pathname = usePathname();

  const navItems = [
    { path: "/", label: "Home", icon: Rocket },
    { path: "/train", label: "Train Model", icon: Brain },
    { path: "/predict", label: "Predict", icon: Target },
    { path: "/console", label: "Console", icon: Terminal },
    { path: "/about", label: "About", icon: Info },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-border/40 backdrop-blur-lg bg-background/80">
      <div className="container mx-auto px-4 py-2 flex items-center justify-between">
        <a
          href="https://www.spaceappschallenge.org/"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center"
        >
          <img src="/SpaceAppsLogo.svg" alt="Logo" className="h-16 w-auto" />
        </a>

        <div className="flex items-center gap-2">
          <BackendStatus />
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = pathname === item.path;

            return (
              <Link
                key={item.path}
                href={item.path}
                className={`
                  flex items-center gap-2 px-4 py-2 rounded-lg transition-all
                  ${
                    isActive
                      ? "bg-primary text-primary-foreground glow-electric"
                      : "hover:bg-muted text-muted-foreground hover:text-foreground"
                  }
                `}
              >
                <Icon className="w-4 h-4" />
                <span className="hidden sm:inline font-medium">
                  {item.label}
                </span>
              </Link>
            );
          })}
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
