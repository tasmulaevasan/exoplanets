"use client";

import { ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster as Sonner } from "sonner";
import "@/app/globals.css";

const queryClient = new QueryClient();

interface ProvidersProps {
  children: ReactNode;
}

const Providers = ({ children }: { children: ReactNode }) => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Sonner />
      {children}
    </TooltipProvider>
  </QueryClientProvider>
);

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="icon" href="/SpaceAppsLogoSmall.svg" type="image/svg+xml" />
        <title>Astrelis: AI</title>
      </head>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
