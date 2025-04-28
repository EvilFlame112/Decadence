"use client"

import { useRef, useEffect, useState } from "react"
import { motion } from "framer-motion"
import { Canvas } from "@react-three/fiber"
import { PresentationControls, useGLTF, Environment } from "@react-three/drei"
import { Button } from "@/components/ui/button"
import { ChevronDown } from "lucide-react"

function Model(props: any) {
  const { scene } = useGLTF("/assets/3d/duck.glb")

  // Clone the scene to avoid modifying the cached original
  const clonedScene = scene.clone()

  // Find all materials and modify them
  clonedScene.traverse((node: any) => {
    if (node.isMesh && node.material) {
      node.material = node.material.clone()
      node.material.color.set("#eab308")
      node.material.emissive.set("#ca8a04")
      node.material.emissiveIntensity = 0.2
      node.material.metalness = 0.8
      node.material.roughness = 0.2
    }
  })

  return <primitive object={clonedScene} {...props} />
}

export default function Hero() {
  const [scrollY, setScrollY] = useState(0)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleScroll = () => {
      setScrollY(window.scrollY)
    }

    window.addEventListener("scroll", handleScroll, { passive: true })
    return () => window.removeEventListener("scroll", handleScroll)
  }, [])

  return (
    <section ref={containerRef} className="relative h-screen w-full overflow-hidden">
      {/* Background elements with parallax */}
      <div
        className="absolute inset-0 bg-gradient-radial from-purple-900/20 via-background to-background z-0 parallax"
        style={{
          transform: `translateY(${scrollY * 0.1}px)`,
        }}
      />

      {/* Subtle grid pattern */}
      <div
        className="absolute inset-0 diagonal-grid opacity-30 z-0 parallax"
        style={{
          transform: `translateY(${scrollY * 0.05}px)`,
        }}
      />

      {/* Animated particles */}
      <div className="absolute inset-0 z-0">
        {Array.from({ length: 20 }).map((_, i) => (
          <motion.div
            key={i}
            className="absolute rounded-full bg-secondary/20"
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

      {/* 3D Model */}
      <div className="absolute inset-0 z-10">
        <Canvas camera={{ position: [0, 0, 5], fov: 45 }}>
          <ambientLight intensity={0.5} />
          <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} intensity={1} castShadow />
          <PresentationControls
            global
            rotation={[0.13, 0.1, 0]}
            polar={[-0.4, 0.2]}
            azimuth={[-0.5, 0.5]}
            config={{ mass: 2, tension: 400 }}
            snap={{ mass: 4, tension: 400 }}
          >
            <Model position={[0, -1, 0]} scale={2} rotation={[0, Math.PI / 4, 0]} />
          </PresentationControls>
          <Environment preset="city" />
        </Canvas>
      </div>

      {/* Content */}
      <div className="relative z-20 flex flex-col items-center justify-center h-full px-4 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.5 }}
          className="max-w-3xl mx-auto parallax"
          style={{
            transform: `translateY(${scrollY * -0.2}px)`,
          }}
        >
          <h1 className="font-serif text-6xl md:text-8xl font-light mb-6 gradient-text glow-text tracking-wide">
            DECADENCE
          </h1>

          <p className="text-xl md:text-2xl mb-12 text-muted-foreground font-light max-w-xl mx-auto tracking-wide">
            Advanced frame interpolation for exquisite video enhancement
          </p>

          <div className="flex flex-col sm:flex-row gap-6 justify-center mt-8">
            <Button
              className="bg-primary hover:bg-primary/90 text-primary-foreground px-8 py-6 text-lg"
              onClick={() => document.getElementById("features")?.scrollIntoView({ behavior: "smooth" })}
            >
              Discover
            </Button>

            <Button
              variant="outline"
              className="border-secondary/30 text-secondary hover:bg-secondary/10 px-8 py-6 text-lg"
              onClick={() => document.getElementById("download")?.scrollIntoView({ behavior: "smooth" })}
            >
              Download
            </Button>
          </div>
        </motion.div>

        <motion.div
          className="absolute bottom-10 left-1/2 transform -translate-x-1/2"
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Number.POSITIVE_INFINITY, ease: "easeInOut" }}
        >
          <Button
            variant="ghost"
            size="icon"
            className="text-muted-foreground hover:text-secondary"
            onClick={() => document.getElementById("features")?.scrollIntoView({ behavior: "smooth" })}
          >
            <ChevronDown />
          </Button>
        </motion.div>
      </div>
    </section>
  )
}

