"use client"

import { useRef, useEffect, useState } from "react"
import { motion } from "framer-motion"
import { useInView } from "react-intersection-observer"

export default function ProcessSection() {
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
    <section
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
        className="absolute inset-0 bg-gradient-to-b from-background via-burgundy-950/10 to-background z-0"
        style={{
          transform: `translateY(${(scrollY - (containerRef.current?.offsetTop || 0)) * 0.1}px)`,
        }}
      />

      {/* Animated lines */}
      <div className="absolute inset-0 overflow-hidden z-0">
        {[...Array(5)].map((_, i) => (
          <motion.div
            key={i}
            className="absolute h-px bg-burgundy-500/10"
            style={{
              top: `${20 + i * 15}%`,
              left: 0,
              right: 0,
              transform: `translateY(${(scrollY - (containerRef.current?.offsetTop || 0)) * 0.05 * (i + 1)}px)`,
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

      <div
        className="absolute inset-0 opacity-5 z-0"
        style={{
          backgroundImage:
            "radial-gradient(circle at 25% 25%, rgba(237, 168, 185, 0.2) 0%, transparent 15%), radial-gradient(circle at 75% 75%, rgba(237, 168, 185, 0.2) 0%, transparent 15%)",
          transform: `translateY(${(scrollY - (containerRef.current?.offsetTop || 0)) * 0.05}px)`,
        }}
      />

      <div className="container relative z-10 px-4 md:px-6">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.8 }}
          className="text-center mb-24"
          style={{
            transform: `translateY(${(scrollY - (containerRef.current?.offsetTop || 0)) * -0.1}px)`,
          }}
        >
          <h2 className="font-serif text-3xl md:text-5xl font-light mb-6 text-secondary">The Process</h2>
          <div className="w-24 h-px bg-gold-400 mx-auto mb-8"></div>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto font-light">
            A sophisticated approach to video enhancement
          </p>
        </motion.div>

        <div className="max-w-4xl mx-auto">
          <div className="space-y-20">
            {steps.map((step, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -50 }}
                animate={inView ? { opacity: 1, x: 0 } : {}}
                transition={{ duration: 0.8, delay: index * 0.2 }}
                className="flex flex-col md:flex-row gap-8 items-start"
                style={{
                  transform: `translateY(${(scrollY - (containerRef.current?.offsetTop || 0) - 300) * -0.05 * (index + 1)}px)`,
                }}
              >
                <motion.div
                  className="flex-shrink-0"
                  whileHover={{
                    rotate: [0, 5, -5, 0],
                    transition: { duration: 0.5 },
                  }}
                >
                  <motion.div
                    className="w-16 h-16 rounded-full bg-burgundy-950/50 flex items-center justify-center text-gold-400 font-serif font-bold text-2xl border border-gold-400/20"
                    animate={{
                      boxShadow: [
                        "0 0 0 rgba(229, 197, 90, 0)",
                        "0 0 20px rgba(229, 197, 90, 0.3)",
                        "0 0 0 rgba(229, 197, 90, 0)",
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
                  <h3 className="text-2xl font-serif font-light text-secondary mb-4">{step.title}</h3>
                  <p className="text-muted-foreground text-lg">{step.description}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  )
}

