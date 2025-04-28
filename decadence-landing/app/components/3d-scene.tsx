"use client"

import { useRef, useState, useEffect } from "react"
import { motion } from "framer-motion"
import Image from "next/image"

export default function LogoScene() {
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!containerRef.current) return

      const rect = containerRef.current.getBoundingClientRect()
      const x = (e.clientX - rect.left) / rect.width
      const y = (e.clientY - rect.top) / rect.height

      setMousePosition({
        x: x - 0.5, // Center at 0
        y: y - 0.5, // Center at 0
      })
    }

    window.addEventListener("mousemove", handleMouseMove)

    return () => {
      window.removeEventListener("mousemove", handleMouseMove)
    }
  }, [])

  return (
    <div ref={containerRef} className="w-full h-full flex items-center justify-center relative">
      {/* Background glow */}
      <div className="absolute w-full h-full bg-gradient-radial from-amber-900/20 via-transparent to-transparent opacity-50" />

      {/* Grid pattern with parallax */}
      <motion.div
        className="absolute inset-0 opacity-10"
        style={{
          backgroundImage:
            "linear-gradient(to right, rgba(217, 119, 6, 0.2) 1px, transparent 1px), linear-gradient(to bottom, rgba(217, 119, 6, 0.2) 1px, transparent 1px)",
          backgroundSize: "30px 30px",
          transform: `translate(${mousePosition.x * -20}px, ${mousePosition.y * -20}px)`,
        }}
      />

      {/* Animated particles */}
      <div className="absolute inset-0">
        {Array.from({ length: 15 }).map((_, i) => (
          <motion.div
            key={i}
            className="absolute rounded-full bg-amber-500/20"
            style={{
              width: Math.random() * 4 + 2,
              height: Math.random() * 4 + 2,
              top: `${Math.random() * 100}%`,
              left: `${Math.random() * 100}%`,
            }}
            animate={{
              y: [0, -30, 0],
              opacity: [0.2, 0.5, 0.2],
            }}
            transition={{
              duration: Math.random() * 5 + 10,
              repeat: Number.POSITIVE_INFINITY,
              ease: "easeInOut",
              delay: Math.random() * 5,
            }}
          />
        ))}
      </div>

      {/* Logo container with animations - positioned higher in the container */}
      <motion.div
        className="relative z-10 mt-[-150px]" // Increased from -120px to -150px to move higher
        animate={{
          y: [0, -10, 0],
        }}
        transition={{
          duration: 6,
          repeat: Number.POSITIVE_INFINITY,
          ease: "easeInOut",
        }}
        style={{
          transform: `translate(${mousePosition.x * -15}px, ${mousePosition.y * -15}px)`,
        }}
      >
        {/* Glow effect behind logo */}
        <motion.div
          className="absolute -inset-10 rounded-full bg-amber-500/10 blur-xl"
          animate={{
            scale: [1, 1.2, 1],
            opacity: [0.5, 0.8, 0.5],
          }}
          transition={{
            duration: 4,
            repeat: Number.POSITIVE_INFINITY,
            ease: "easeInOut",
          }}
        />

        {/* Logo with hover effect */}
        <motion.div
          whileHover={{
            scale: 1.1,
            rotate: [0, -5, 5, 0],
            transition: { duration: 0.5 },
          }}
          className="flex flex-col items-center"
        >
          <Image
            src="/images/logo_no_bg.ico"
            alt="Decadence Logo"
            width={180}
            height={180}
            className="drop-shadow-[0_0_15px_rgba(234,179,8,0.5)]"
          />

          {/* Text below logo */}
          <motion.div
            className="mt-8 text-center"
            animate={{
              y: [0, -5, 0],
              opacity: [0.7, 1, 0.7],
            }}
            transition={{
              duration: 3,
              repeat: Number.POSITIVE_INFINITY,
              ease: "easeInOut",
            }}
          >
            <h2 className="text-3xl font-light text-amber-500 mb-2 tracking-widest">DECADENCE</h2>
            <p className="text-amber-400/70 tracking-wider">FRAME INTERPOLATION</p>
          </motion.div>
        </motion.div>
      </motion.div>
    </div>
  )
}

