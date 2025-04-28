"use client"

import { useRef, useState, useEffect } from "react"
import { Canvas, useFrame, useThree } from "@react-three/fiber"
import { useGLTF, OrbitControls, Environment, Float } from "@react-three/drei"
import { motion } from "framer-motion"
import { Button } from "@/components/ui/button"
import { ArrowDown } from "lucide-react"
import type * as THREE from "three"

function Model(props: any) {
  const { nodes, materials } = useGLTF("/assets/3d/duck.glb") as any
  const mesh = useRef<THREE.Mesh>(null)
  const { viewport } = useThree()
  const [hovered, setHovered] = useState(false)

  useFrame((state) => {
    if (!mesh.current) return
    mesh.current.rotation.y = state.clock.getElapsedTime() * 0.15
  })

  return (
    <Float speed={2} rotationIntensity={0.2} floatIntensity={0.5}>
      <mesh
        {...props}
        ref={mesh}
        scale={viewport.width < 5 ? 1.5 : 2}
        onPointerOver={() => setHovered(true)}
        onPointerOut={() => setHovered(false)}
        geometry={nodes.LOD3spShape.geometry}
        material={materials.blinn3}
        material-color={hovered ? "#e5c55a" : "#ddb333"}
        material-roughness={0.3}
        material-metalness={0.8}
        position={[0, -1, 0]}
      />
    </Float>
  )
}

export default function HeroScene() {
  const [scrollY, setScrollY] = useState(0)

  useEffect(() => {
    const handleScroll = () => {
      setScrollY(window.scrollY)
    }

    window.addEventListener("scroll", handleScroll, { passive: true })
    return () => window.removeEventListener("scroll", handleScroll)
  }, [])

  return (
    <section className="relative h-screen w-full overflow-hidden bg-background">
      {/* Background elements with parallax */}
      <div
        className="absolute inset-0 bg-gradient-radial from-burgundy-900/20 via-background to-background z-0"
        style={{
          transform: `translateY(${scrollY * 0.1}px)`,
        }}
      />

      <div
        className="absolute top-0 left-0 right-0 h-32 bg-gradient-to-b from-burgundy-950/50 to-transparent z-0"
        style={{
          transform: `translateY(${scrollY * 0.2}px)`,
        }}
      />

      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-background to-transparent z-0" />

      {/* Subtle grid pattern */}
      <div
        className="absolute inset-0 opacity-5 z-0"
        style={{
          backgroundImage:
            "linear-gradient(to right, rgba(237, 168, 185, 0.1) 1px, transparent 1px), linear-gradient(to bottom, rgba(237, 168, 185, 0.1) 1px, transparent 1px)",
          backgroundSize: "80px 80px",
          transform: `translateY(${scrollY * 0.05}px)`,
        }}
      />

      {/* 3D Scene */}
      <div className="absolute inset-0 z-10">
        <Canvas camera={{ position: [0, 0, 6], fov: 40 }}>
          <ambientLight intensity={0.5} />
          <spotLight position={[10, 10, 10]} angle={0.15} penumbra={1} intensity={1} castShadow />
          <Model />
          <Environment preset="studio" />
          <OrbitControls
            enableZoom={false}
            enablePan={false}
            minPolarAngle={Math.PI / 2.5}
            maxPolarAngle={Math.PI / 1.8}
            rotateSpeed={0.5}
          />
        </Canvas>
      </div>

      {/* Content */}
      <div className="relative z-20 flex flex-col items-center justify-center h-full px-4 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.5 }}
          className="max-w-3xl mx-auto"
          style={{
            transform: `translateY(${scrollY * -0.2}px)`,
          }}
        >
          <h1 className="font-serif text-5xl md:text-7xl font-light mb-6 text-shadow-lg">
            <span className="text-secondary">Decadence</span>
          </h1>

          <p className="text-xl md:text-2xl mb-8 text-muted-foreground font-light max-w-xl mx-auto text-shadow-sm">
            Advanced frame interpolation for exquisite video enhancement
          </p>

          <div className="flex flex-col sm:flex-row gap-6 justify-center mt-8">
            <Button
              className="burgundy-gradient hover:opacity-90 transition-opacity text-secondary px-8 py-6 text-lg"
              onClick={() => document.getElementById("features")?.scrollIntoView({ behavior: "smooth" })}
            >
              Discover
            </Button>

            <Button
              variant="outline"
              className="border-gold-400/30 text-gold-400 hover:bg-gold-400/10 px-8 py-6 text-lg"
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
            <ArrowDown />
          </Button>
        </motion.div>
      </div>
    </section>
  )
}

