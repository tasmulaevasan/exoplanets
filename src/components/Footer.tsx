import { Github, Users } from "lucide-react";
import Link from "next/link";

const Footer = () => {
  return (
    <footer className="relative mt-auto border-t border-border/40 backdrop-blur-lg bg-background/80">
      <div className="container mx-auto px-4 py-8">
        <div className="flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="text-center md:text-left">
            <h3 className="font-display font-bold text-lg gradient-text">
              Astrelis
            </h3>
            <p className="text-sm text-muted-foreground mt-1">Exoplanet AI</p>
            <p className="text-sm text-muted-foreground mt-1">
              NASA Space Apps Challenge 2025
            </p>
          </div>

          <div className="flex items-center gap-4">
            <Link
              href="https://github.com/tasmulaevasan/exoplanets/tree/main"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2 rounded-lg border border-border/40 hover:border-primary/50 hover:bg-primary/10 transition-all text-sm"
            >
              <Github className="w-4 h-4" />
              <span className="hidden sm:inline">GitHub</span>
            </Link>
            <Link
              href="https://www.spaceappschallenge.org/2025/find-a-team/astrelis/"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 px-4 py-2 rounded-lg border border-border/40 hover:border-primary/50 hover:bg-primary/10 transition-all text-sm"
            >
              <Users className="w-4 h-4" />
              <span className="hidden sm:inline">Our Team</span>
            </Link>
          </div>
        </div>

        <div className="text-center text-xs text-muted-foreground mt-6 pt-6 border-t border-border/20">
          <p>© 2025 Astrelis. Built for Space Apps Challenge.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
