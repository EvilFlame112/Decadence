"use client"
import { motion } from "framer-motion"

interface ParticlesProps {
  className?: string
  quantity?: number
  color?: string
  stationary?: boolean
}

export default function Particles({
  className = "",
  quantity = 50,
  color = "#ffffff",
  stationary = false,
}: ParticlesProps) {
  const particles = Array.from({ length: quantity })

  return (
    <div className={`absolute inset-0 ${className}`}>
      {particles.map((_, i) => (
        <Particle key={i} color={color} stationary={stationary} />
      ))}
    </div>
  )
}

function Particle({ color, stationary }: { color: string; stationary: boolean }) {
  const size = Math.random() * 3 + 1
  const initialX = Math.random() * 100
  const initialY = Math.random() * 100
  const duration = Math.random() * 20 + 10
  const delay = Math.random() * 5

  if (stationary) {
    return (
      <motion.div
        className="absolute rounded-full"
        style={{
          width: `${size}px`,
          height: `${size}px`,
          backgroundColor: color,
          top: `${initialY}%`,
          left: `${initialX}%`,
        }}
        animate={{
          opacity: [0.1, 0.5, 0.1],
        }}
        transition={{
          duration: duration / 2,
          repeat: Number.POSITIVE_INFINITY,
          ease: "easeInOut",
          delay,
        }}
      />
    )
  }

  return (
    <motion.div
      className="absolute rounded-full"
      style={{
        width: `${size}px`,
        height: `${size}px`,
        backgroundColor: color,
        top: `${initialY}%`,
        left: `${initialX}%`,
      }}
      animate={{
        y: [0, -30, 0],
        x: [0, Math.random() * 20 - 10, 0],
        opacity: [0.1, 0.5, 0.1],
      }}
      transition={{
        duration,
        repeat: Number.POSITIVE_INFINITY,
        ease: "easeInOut",
        delay,
      }}
    />
  )
}

