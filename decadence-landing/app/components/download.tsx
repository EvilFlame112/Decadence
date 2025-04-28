"use client"

import { useRef } from "react"
import { motion, useScroll, useTransform } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Download, Github } from "lucide-react"

export default function DownloadSection() {
  const containerRef = useRef<HTMLDivElement>(null)
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end start"],
  })

  const y1 = useTransform(scrollYProgress, [0, 1], [0, -100])
  const y2 = useTransform(scrollYProgress, [0, 1], [0, -200])

  return (
    <section id="download" ref={containerRef} className="relative py-32 overflow-hidden">
      {/* Background elements with parallax */}
      <motion.div
        className="absolute inset-0 bg-gradient-radial from-purple-900/20 via-background to-background z-0"
        style={{ y: y1 }}
      />

      <motion.div className="absolute inset-0 diagonal-grid opacity-30 z-0" style={{ y: y2 }} />

      <div className="container relative z-10 px-4 md:px-6">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true, margin: "-100px" }}
          className="text-center mb-24 max-w-3xl mx-auto"
        >
          <h2 className="font-serif text-4xl md:text-6xl font-light mb-6 gradient-text glow-text tracking-wide">
            DOWNLOAD DECADENCE
          </h2>
          <div className="w-24 h-px bg-secondary mx-auto mb-8"></div>
          <p className="text-xl text-muted-foreground font-light tracking-wide">
            Transform your videos with professional-grade frame interpolation
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.3 }}
          viewport={{ once: true, margin: "-100px" }}
          whileHover={{ y: -10 }}
          className="max-w-md mx-auto mb-16"
        >
          <div className="p-8 border border-secondary/20 rounded-lg bg-card/50 backdrop-blur-sm glow gradient-border">
            <h3 className="font-serif text-2xl font-light text-secondary mb-6 text-center tracking-wide">WINDOWS</h3>

            <div className="flex justify-between text-sm text-muted-foreground mb-6">
              <span>Version 1.2.0</span>
              <span>64-bit</span>
            </div>

            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button className="w-full bg-primary hover:bg-primary/90 text-primary-foreground py-6 text-lg">
                <Download className="mr-3 h-5 w-5" />
                DOWNLOAD .EXE
              </Button>
            </motion.div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.5 }}
          viewport={{ once: true, margin: "-100px" }}
          className="flex justify-center"
        >
          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Button
              variant="outline"
              size="lg"
              className="border-secondary/30 text-secondary hover:bg-secondary/10 px-8 py-6 text-lg"
            >
              <Github className="mr-3 h-5 w-5" />
              VIEW ON GITHUB
            </Button>
          </motion.div>
        </motion.div>
      </div>
    </section>
  )
}

