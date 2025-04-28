"use client"

import { useRef } from "react"
import { motion, useScroll, useTransform } from "framer-motion"

export default function Process() {
  const containerRef = useRef<HTMLDivElement>(null)
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end start"],
  })

  const y1 = useTransform(scrollYProgress, [0, 1], [0, -100])
  const y2 = useTransform(scrollYProgress, [0, 1], [0, -200])

  const steps = [
    {
      number: "01",
      title: "Frame Analysis",
      description:
        "Decadence analyzes each frame of your video to understand motion patterns and object relationships.",
    },
    {
      number: "02",
      title: "AI Prediction",
      description: "The neural network predicts how objects move between frames and generates new intermediate frames.",
    },
    {
      number: "03",
      title: "Quality Enhancement",
      description: "Advanced algorithms refine the generated frames to ensure consistency and visual quality.",
    },
    {
      number: "04",
      title: "Output Generation",
      description: "The final video is assembled with the new frames integrated for smooth, fluid motion.",
    },
  ]

  return (
    <section ref={containerRef} className="relative py-32 overflow-hidden">
      {/* Background elements with parallax */}
      <motion.div
        className="absolute inset-0 bg-gradient-to-b from-background via-purple-950/10 to-background z-0"
        style={{ y: y1 }}
      />

      {/* Animated lines */}
      <div className="absolute inset-0 overflow-hidden z-0">
        {[...Array(5)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute h-px bg-secondary/10"
            style={{
              top: `${20 + i * 15}%`,
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

      <motion.div className="absolute inset-0 vertical-lines opacity-30 z-0" style={{ y: y2 }} />

      <div className="container relative z-10 px-4 md:px-6">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true, margin: "-100px" }}
          className="text-center mb-24"
        >
          <h2 className="font-serif text-4xl md:text-6xl font-light mb-6 gradient-text glow-text tracking-wide">
            THE PROCESS
          </h2>
          <div className="w-24 h-px bg-secondary mx-auto mb-8"></div>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto font-light tracking-wide">
            A sophisticated approach to video enhancement
          </p>
        </motion.div>

        <div className="max-w-4xl mx-auto">
          <div className="space-y-20">
            {steps.map((step, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -50 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.8, delay: index * 0.2 }}
                viewport={{ once: true, margin: "-100px" }}
                className="flex flex-col md:flex-row gap-8 items-start"
              >
                <motion.div
                  className="flex-shrink-0"
                  whileHover={{
                    rotate: [0, 5, -5, 0],
                    transition: { duration: 0.5 },
                  }}
                >
                  <motion.div
                    className="w-16 h-16 rounded-full bg-primary/20 flex items-center justify-center text-secondary font-serif font-bold text-2xl border border-secondary/20"
                    animate={{
                      boxShadow: [
                        "0 0 0 rgba(234, 179, 8, 0)",
                        "0 0 20px rgba(234, 179, 8, 0.3)",
                        "0 0 0 rgba(234, 179, 8, 0)",
                      ],
                    }}
                    transition={{
                      duration: 3,
                      repeat: Number.POSITIVE_INFINITY,
                      delay: index * 0.5,
                    }}
                  >
                    {step.number}
                  </motion.div>
                </motion.div>
                <div>
                  <h3 className="text-2xl font-serif font-light text-secondary mb-4 tracking-wide">{step.title}</h3>
                  <p className="text-muted-foreground text-lg tracking-wide">{step.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

