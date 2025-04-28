"use client"

import { useRef, useEffect, useState } from "react"
import { motion } from "framer-motion"
import { useInView } from "react-intersection-observer"
import { Button } from "@/components/ui/button"
import { Download, Github } from "lucide-react"

export default function DownloadSection() {
  const [scrollY, setScrollY] = useState(0)
  const containerRef = useRef<HTMLDivElement>(null)
  const [ref, inView] = useInView({
    triggerOnce: false,
    threshold: 0.1,
  })

  useEffect(() => {
    const handleScroll = () => {
      setScrollY(window.scrollY)
    }

    window.addEventListener("scroll", handleScroll, { passive: true })
    return () => window.removeEventListener("scroll", handleScroll)
  }, [])

  return (
    <section
      id="download"
      ref={(el) => {
        containerRef.current = el
        if (ref) {
          ref(el)
        }
      }}
      className="relative py-32 overflow-hidden"
    >
      {/* Background elements with parallax */}
      <div
        className="absolute inset-0 bg-gradient-radial from-burgundy-950/20 via-background to-background z-0"
        style={{
          transform: `translateY(${(scrollY - (containerRef.current?.offsetTop || 0)) * 0.1}px)`,
        }}
      />

      <div
        className="absolute inset-0 opacity-5 z-0"
        style={{
          backgroundImage:
            "linear-gradient(45deg, rgba(229, 197, 90, 0.1) 1px, transparent 1px), linear-gradient(135deg, rgba(229, 197, 90, 0.1) 1px, transparent 1px)",
          backgroundSize: "30px 30px",
          transform: `translateY(${(scrollY - (containerRef.current?.offsetTop || 0)) * 0.05}px)`,
        }}
      />

      <div className="container relative z-10 px-4 md:px-6">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
          className="text-center mb-24 max-w-3xl mx-auto"
          style={{
            transform: `translateY(${(scrollY - (containerRef.current?.offsetTop || 0)) * -0.1}px)`,
          }}
        >
          <h2 className="font-serif text-3xl md:text-5xl font-light mb-6 text-secondary">Download Decadence</h2>
          <div className="w-24 h-px bg-gold-400 mx-auto mb-8"></div>
          <p className="text-xl text-muted-foreground font-light">
            Transform your videos with professional-grade frame interpolation
          </p>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8, delay: 0.3 }}
          className="max-w-md mx-auto mb-16"
          style={{
            transform: `translateY(${(scrollY - (containerRef.current?.offsetTop || 0) - 300) * -0.05}px)`,
          }}
        >
          <div className="p-8 border border-gold-400/20 rounded-lg bg-card/50 backdrop-blur-sm elegant-shadow">
            <h3 className="font-serif text-2xl font-light text-secondary mb-6 text-center">Windows</h3>

            <div className="flex justify-between text-sm text-muted-foreground mb-6">
              <span>Version 1.2.0</span>
              <span>64-bit</span>
            </div>

            <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
              <Button className="w-full burgundy-gradient hover:opacity-90 transition-opacity text-secondary py-6 text-lg">
                <Download className="mr-3 h-5 w-5" />
                Download .exe
              </Button>
            </motion.div>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={inView ? { opacity: 1 } : {}}
          transition={{ duration: 0.8, delay: 0.5 }}
          className="flex justify-center"
        >
          <motion.div whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}>
            <Button
              variant="outline"
              size="lg"
              className="border-gold-400/30 text-gold-400 hover:bg-gold-400/10 px-8 py-6 text-lg"
            >
              <Github className="mr-3 h-5 w-5" />
              View on GitHub
            </Button>
          </motion.div>
        </motion.div>
      </div>
    </section>
  )
}

