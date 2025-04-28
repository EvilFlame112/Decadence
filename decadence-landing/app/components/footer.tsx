"use client"

import { motion } from "framer-motion"
import { Github } from "lucide-react"

export default function Footer() {
  return (
    <footer className="py-16 bg-card/50 text-muted-foreground relative overflow-hidden">
      {/* Animated background lines */}
      <div className="absolute inset-0">
        {[...Array(3)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute h-px bg-secondary/10"
            style={{
              top: `${30 + i * 20}%`,
              left: 0,
              right: 0,
            }}
            animate={{
              x: ["-100%", "100%"],
            }}
            transition={{
              duration: 20 + i * 5,
              repeat: Number.POSITIVE_INFINITY,
              ease: "linear",
            }}
          />
        ))}
      </div>

      <div className="absolute inset-0 horizontal-lines opacity-30 z-0" />

      <div className="container px-4 md:px-6 relative z-10">
        <motion.div
          className="flex flex-col items-center justify-center mb-12"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
        >
          <h3 className="font-serif text-2xl font-light text-secondary mb-4 tracking-wide">DECADENCE</h3>
          <p className="text-center max-w-md mb-8 tracking-wide">Advanced frame interpolation for video enhancement</p>
          <div className="flex space-x-6">
            <motion.a
              href="#"
              className="text-muted-foreground hover:text-secondary transition-colors"
              whileHover={{
                scale: 1.2,
                rotate: [-5, 5, 0],
                transition: { duration: 0.3 },
              }}
            >
              <Github className="h-6 w-6" />
            </motion.a>
          </div>
        </motion.div>

        <motion.div
          className="border-t border-muted pt-8 text-center"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          viewport={{ once: true }}
        >
          <p className="tracking-wide">© {new Date().getFullYear()} DECADENCE. ALL RIGHTS RESERVED.</p>
        </motion.div>
      </div>
    </footer>
  )
}

