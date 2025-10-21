"use client";

import { motion, AnimatePresence } from "framer-motion";
import { usePathname } from "next/navigation";
import { ReactNode } from "react";

interface PageTransitionProps {
  children: ReactNode;
}

export default function PageTransition({ children }: PageTransitionProps) {
  const pathname = usePathname();

  // Конфигурация полосок
  const stripes = [
    { delay: 0, height: "20%", color: "#0960e1" },
    { delay: 0.12, height: "20%", color: "#1a7de8" },
    { delay: 0.24, height: "20%", color: "#2e96f5" },
    { delay: 0.36, height: "20%", color: "#3fa3f6" },
    { delay: 0.48, height: "20%", color: "#4aa5f8" },
  ];

  return (
    <AnimatePresence mode="wait">
      <motion.div key={pathname} className="relative">
        {stripes.map((stripe, index) => (
          <motion.div
            key={index}
            className="fixed left-0 z-[9999] pointer-events-none"
            initial={{ x: "-100%" }}
            animate={{ x: "100%" }}
            transition={{
              duration: 1.2,
              delay: stripe.delay,
              ease: [0.22, 1, 0.36, 1],
            }}
            style={{
              top: `${index * 20}%`,
              height: stripe.height,
              width: "200%",
              backgroundColor: stripe.color,
              boxShadow: `0 0 30px ${stripe.color}80`,
            }}
          />
        ))}

        {/* Page content */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{
            duration: 0.3,
            delay: 0.4,
            ease: "easeOut",
          }}
        >
          {children}
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}
