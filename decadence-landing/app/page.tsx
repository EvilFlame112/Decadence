"use client"

import { useEffect, useRef, useState } from "react"
import Image from "next/image"
import { motion, useScroll, useTransform, AnimatePresence } from "framer-motion"
import { Download, Github, ArrowRight, Zap, Code, ChevronDown, Film, Clock } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import Particles from "./components/particles"
import { TypeAnimation } from "react-type-animation"
import { useInView } from "react-intersection-observer"
import dynamic from "next/dynamic"

// Dynamically import the logo scene with no SSR to avoid hydration issues
const LogoScene = dynamic(() => import("./components/3d-scene"), {
  ssr: false,
  loading: () => (
    <div className="w-full h-full flex items-center justify-center">
      <div className="flex flex-col items-center justify-center">
        <div className="w-32 h-32 rounded-full bg-amber-900/20 animate-pulse mb-4"></div>
        <div className="text-amber-500 text-2xl font-light animate-pulse">Loading...</div>
      </div>
    </div>
  ),
})

export default function Home() {
  const [scrollY, setScrollY] = useState(0)
  const heroRef = useRef<HTMLDivElement>(null)
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })
  const [is3DLoaded, setIs3DLoaded] = useState(false)

  // Refs for scroll animations with higher thresholds for better timing
  const [featuresRef, featuresInView] = useInView({
    triggerOnce: false,
    threshold: 0.25,
  })

  const [processRef, processInView] = useInView({
    triggerOnce: false,
    threshold: 0.25,
  })

  const [downloadRef, downloadInView] = useInView({
    triggerOnce: false,
    threshold: 0.25,
  })

  // Scroll progress for parallax effects
  const { scrollYProgress: heroScrollProgress } = useScroll({
    target: heroRef,
    offset: ["start start", "end start"],
  })

  const heroOpacity = useTransform(heroScrollProgress, [0, 0.8], [1, 0])
  const heroY = useTransform(heroScrollProgress, [0, 0.8], [0, 100])

  useEffect(() => {
    const handleScroll = () => {
      setScrollY(window.scrollY)
    }

    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({
        x: e.clientX / window.innerWidth,
        y: e.clientY / window.innerHeight,
      })
    }

    window.addEventListener("scroll", handleScroll, { passive: true })
    window.addEventListener("mousemove", handleMouseMove, { passive: true })

    // Set logo loaded immediately
    setIs3DLoaded(true)

    return () => {
      window.removeEventListener("scroll", handleScroll)
      window.removeEventListener("mousemove", handleMouseMove)
    }
  }, [])

  // Animated diamond patterns
  const diamondPatterns = [
    { size: "w-96 h-96", top: "top-1/4", left: "left-1/4", delay: 0 },
    { size: "w-64 h-64", top: "top-1/3", left: "left-1/3", delay: 0.5 },
    { size: "w-32 h-32", top: "top-1/2", left: "left-1/2", delay: 1 },
    { size: "w-48 h-48", top: "top-2/3", left: "left-1/4", delay: 1.5 },
    { size: "w-72 h-72", top: "top-1/4", left: "left-2/3", delay: 2 },
  ]

  // Enhanced fade variants for animations
  const fadeInUpVariants = {
    hidden: { opacity: 0, y: 50 },
    visible: (delay = 0) => ({
      opacity: 1,
      y: 0,
      transition: {
        duration: 0.8,
        delay: delay,
        ease: [0.22, 1, 0.36, 1],
      },
    }),
  }

  const fadeInLeftVariants = {
    hidden: { opacity: 0, x: -50 },
    visible: (delay = 0) => ({
      opacity: 1,
      x: 0,
      transition: {
        duration: 0.8,
        delay: delay,
        ease: [0.22, 1, 0.36, 1],
      },
    }),
  }

  const staggerChildrenVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.2,
      },
    },
  }

  return (
    <main className="min-h-screen bg-black text-white overflow-x-hidden">
      {/* Hero Section with Enhanced Animations and Logo Component */}
      <motion.div
        ref={heroRef}
        className="relative h-screen flex items-center justify-center overflow-hidden"
        style={{ opacity: heroOpacity, y: heroY }}
      >
        {/* Particle background */}
        <Particles className="absolute inset-0 z-0" quantity={100} color="#d97706" />

        {/* Grid pattern overlay with parallax effect */}
        <motion.div
          className="absolute inset-0 z-0 opacity-10"
          style={{
            backgroundImage:
              "linear-gradient(to right, rgba(217, 119, 6, 0.1) 1px, transparent 1px), linear-gradient(to bottom, rgba(217, 119, 6, 0.1) 1px, transparent 1px)",
            backgroundSize: "50px 50px",
            x: mousePosition.x * -20,
            y: mousePosition.y * -20,
          }}
        />

        <motion.div
          className="absolute inset-0 z-0"
          style={{
            background: "radial-gradient(circle at center, rgba(30, 30, 30, 1) 0%, rgba(0, 0, 0, 1) 70%)",
            transform: `scale(${1 + scrollY * 0.0005})`,
          }}
        />

        {/* Animated geometric patterns with enhanced parallax */}
        <div className="absolute inset-0 z-0 opacity-20">
          {diamondPatterns.map((pattern, index) => (
            <motion.div
              key={index}
              className={`absolute ${pattern.top} ${pattern.left} ${pattern.size} border border-amber-500/20 rotate-45 transform-gpu`}
              animate={{
                rotate: [45, 135, 225, 315, 405],
                scale: [1, 1.1, 1, 0.9, 1],
                opacity: [0.2, 0.3, 0.2, 0.3, 0.2],
              }}
              transition={{
                duration: 20,
                ease: "linear",
                repeat: Number.POSITIVE_INFINITY,
                delay: pattern.delay,
              }}
              style={{
                x: mousePosition.x * -30 * (index + 1),
                y: mousePosition.y * -30 * (index + 1),
              }}
            />
          ))}
        </div>

        {/* Logo Scene Component */}
        <div className="absolute inset-0 z-10">{is3DLoaded && <LogoScene />}</div>

        <div className="container relative z-20 px-4 md:px-6">
          <AnimatePresence>
            <motion.div
              initial={{ opacity: 0, y: 50 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{
                duration: 1.2,
                ease: [0.22, 1, 0.36, 1],
                staggerChildren: 0.2,
              }}
              className="text-center max-w-3xl mx-auto"
            >
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{
                  opacity: 1,
                  scale: 1,
                  transition: {
                    duration: 1.5,
                    ease: [0.22, 1, 0.36, 1],
                  },
                }}
                className="h-12 mb-10 mt-[350px]" // Increased top margin, reduced bottom margin
              >
                <TypeAnimation
                  sequence={[
                    "ADVANCED FRAME INTERPOLATION",
                    2000,
                    "TRANSFORM YOUR VIDEOS",
                    2000,
                    "FLUID MOTION ENHANCEMENT",
                    2000,
                  ]}
                  wrapper="p"
                  speed={50}
                  repeat={Number.POSITIVE_INFINITY}
                  className="text-xl md:text-2xl text-gray-300 font-light tracking-widest"
                />
              </motion.div>

              <motion.div
                className="flex flex-col sm:flex-row gap-6 justify-center mt-4" // Reduced margin-top
                initial={{ opacity: 0, y: 30 }}
                animate={{
                  opacity: 1,
                  y: 0,
                  transition: {
                    duration: 0.8,
                    delay: 1,
                    ease: [0.22, 1, 0.36, 1],
                  },
                }}
              >
                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  transition={{ type: "spring", stiffness: 400, damping: 17 }}
                >
                  <Button
                    size="lg"
                    className="bg-gradient-to-r from-amber-600 to-yellow-600 hover:from-amber-700 hover:to-yellow-700 px-8 py-6 text-lg relative overflow-hidden group tracking-wider"
                    onClick={() => document.getElementById("download")?.scrollIntoView({ behavior: "smooth" })}
                  >
                    <span className="absolute inset-0 w-full h-full transition-all duration-300 ease-out bg-white opacity-0 group-hover:opacity-10"></span>
                    <Download className="mr-3 h-5 w-5" />
                    DOWNLOAD
                  </Button>
                </motion.div>

                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  transition={{ type: "spring", stiffness: 400, damping: 17 }}
                >
                  <Button
                    variant="outline"
                    size="lg"
                    className="border-amber-600/50 hover:bg-amber-600/10 px-8 py-6 text-lg tracking-wider"
                    onClick={() => document.getElementById("features")?.scrollIntoView({ behavior: "smooth" })}
                  >
                    DISCOVER
                    <ArrowRight className="ml-3 h-5 w-5" />
                  </Button>
                </motion.div>
              </motion.div>
            </motion.div>
          </AnimatePresence>
        </div>

        <motion.div
          animate={{
            y: [0, 10, 0],
          }}
          transition={{
            duration: 2,
            repeat: Number.POSITIVE_INFINITY,
            ease: "easeInOut",
          }}
          className="absolute bottom-10 left-1/2 transform -translate-x-1/2 z-10 text-amber-500/70"
        >
          <ChevronDown className="h-8 w-8" />
        </motion.div>

        <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-black to-transparent z-10" />
      </motion.div>

      {/* Features Section */}
      <section id="features" className="py-32 bg-black relative" ref={featuresRef}>
        {/* Animated background gradient with parallax */}
        <motion.div
          className="absolute inset-0 z-0 opacity-10"
          style={{
            background:
              "radial-gradient(circle at 30% 30%, rgba(217, 119, 6, 0.4) 0%, transparent 30%), radial-gradient(circle at 70% 70%, rgba(202, 138, 4, 0.4) 0%, transparent 30%)",
            x: mousePosition.x * -20,
            y: mousePosition.y * -20,
          }}
        />

        {/* Tech grid pattern with parallax */}
        <motion.div
          className="absolute inset-0 z-0 opacity-5"
          style={{
            backgroundImage:
              "linear-gradient(to right, rgba(217, 119, 6, 0.5) 1px, transparent 1px), linear-gradient(to bottom, rgba(217, 119, 6, 0.5) 1px, transparent 1px)",
            backgroundSize: "30px 30px",
            x: mousePosition.x * -10,
            y: mousePosition.y * -10,
          }}
        />

        <div className="container px-4 md:px-6 relative z-10">
          <motion.div
            className="text-center mb-24"
            variants={fadeInUpVariants}
            initial="hidden"
            animate={featuresInView ? "visible" : "hidden"}
            custom={0}
          >
            <h2 className="text-3xl md:text-5xl font-light mb-6 bg-clip-text text-transparent bg-gradient-to-r from-amber-400 to-yellow-600 tracking-wider">
              POWERFUL FEATURES
            </h2>
            <motion.div
              className="w-24 h-1 bg-gradient-to-r from-amber-400 to-yellow-600 mx-auto mb-6"
              initial={{ width: 0 }}
              animate={featuresInView ? { width: 96 } : { width: 0 }}
              transition={{ duration: 1, delay: 0.3, ease: [0.22, 1, 0.36, 1] }}
            ></motion.div>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto font-light tracking-wide">
              Professional-grade frame interpolation with elegant simplicity
            </p>
          </motion.div>

          <motion.div
            className="grid grid-cols-1 md:grid-cols-2 gap-16"
            variants={staggerChildrenVariants}
            initial="hidden"
            animate={featuresInView ? "visible" : "hidden"}
          >
            {[
              {
                title: "AI-POWERED INTERPOLATION",
                description:
                  "Neural network models analyze motion patterns to create natural-looking intermediate frames",
                icon: <Zap className="h-10 w-10 text-amber-400" />,
                delay: 0.1,
              },
              {
                title: "CUSTOMIZABLE SETTINGS",
                description: "Fine-tune interpolation parameters to achieve your desired visual style and performance",
                icon: <Code className="h-10 w-10 text-yellow-400" />,
                delay: 0.3,
              },
            ].map((feature, index) => (
              <motion.div
                key={index}
                variants={fadeInUpVariants}
                custom={feature.delay}
                whileHover={{
                  y: -10,
                  boxShadow: "0 20px 25px -5px rgba(217, 119, 6, 0.1), 0 10px 10px -5px rgba(217, 119, 6, 0.04)",
                  transition: { type: "spring", stiffness: 300, damping: 15 },
                }}
                className="rounded-lg"
              >
                <Card className="bg-transparent border-amber-900/20 backdrop-blur-sm h-full hover:bg-amber-900/10 hover:border-amber-600/40 transition-all duration-500 group">
                  <CardHeader>
                    <motion.div
                      className="mb-6 p-4 rounded-full w-fit bg-amber-900/10"
                      whileHover={{
                        rotate: [0, 10, -10, 0],
                        scale: 1.1,
                        transition: { duration: 0.5 },
                      }}
                    >
                      {feature.icon}
                    </motion.div>
                    <CardTitle className="text-2xl text-white tracking-wide font-light">{feature.title}</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <CardDescription className="text-gray-400 text-lg tracking-wide">
                      {feature.description}
                    </CardDescription>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </motion.div>

          {/* Experience the Difference - Two Static Cards with enhanced animations */}
          <div className="mt-32 max-w-5xl mx-auto">
            <motion.div
              className="text-center mb-16"
              variants={fadeInUpVariants}
              initial="hidden"
              animate={featuresInView ? "visible" : "hidden"}
              custom={0.6}
            >
              <h3 className="text-2xl md:text-3xl font-light mb-4 text-white tracking-wider">
                EXPERIENCE THE DIFFERENCE
              </h3>
              <p className="text-gray-400 tracking-wide">See how Decadence transforms your videos</p>
            </motion.div>

            <motion.div
              className="grid grid-cols-1 md:grid-cols-2 gap-8"
              variants={staggerChildrenVariants}
              initial="hidden"
              animate={featuresInView ? "visible" : "hidden"}
            >
              {[
                {
                  title: "24FPS TO 60FPS CONVERSION",
                  description: "Transform standard footage into silky smooth high frame rate video",
                  icon: <Film className="h-8 w-8 text-amber-400" />,
                  delay: 0.7,
                  image: "/placeholder.svg?height=100&width=200",
                },
                {
                  title: "LOW FRAME INTERPOLATION",
                  description: "Interpolate two images to make content worthy videos",
                  icon: <Clock className="h-8 w-8 text-amber-400" />,
                  delay: 0.8,
                  image: "/placeholder.svg?height=100&width=200",
                },
              ].map((feature, index) => (
                <motion.div
                  key={index}
                  variants={fadeInUpVariants}
                  custom={feature.delay}
                  whileHover={{
                    y: -5,
                    boxShadow: "0 20px 25px -5px rgba(217, 119, 6, 0.1), 0 10px 10px -5px rgba(217, 119, 6, 0.04)",
                    transition: { type: "spring", stiffness: 300, damping: 15 },
                  }}
                >
                  <Card className="bg-transparent border border-amber-600/30 backdrop-blur-sm h-full hover:border-amber-500/80 hover:bg-amber-900/10 transition-all duration-500 overflow-hidden group">
                    <motion.div
                      className="absolute inset-0 z-0 opacity-10"
                      style={{
                        backgroundImage:
                          "radial-gradient(circle at 50% 50%, rgba(217, 119, 6, 0.5) 0%, transparent 70%)",
                        x: mousePosition.x * -20,
                        y: mousePosition.y * -20,
                      }}
                    />
                    <CardHeader className="pb-2 relative z-10">
                      <div className="flex items-center mb-2">
                        <motion.div
                          className="mr-3 p-2 rounded-full bg-amber-900/10"
                          whileHover={{
                            rotate: [0, 10, -10, 0],
                            scale: 1.1,
                            transition: { duration: 0.5 },
                          }}
                        >
                          {feature.icon}
                        </motion.div>
                        <CardTitle className="text-xl text-white tracking-wide font-light">{feature.title}</CardTitle>
                      </div>
                    </CardHeader>
                    <CardContent className="relative z-10">
                      <CardDescription className="text-gray-400 tracking-wide mb-4">
                        {feature.description}
                      </CardDescription>
                      <motion.div
                        className="mt-4 overflow-hidden rounded-md bg-black/30 border border-amber-600/10"
                        whileHover={{
                          scale: 1.02,
                          transition: { duration: 0.3 },
                        }}
                      >
                        <Image
                          src={feature.image || "/placeholder.svg"}
                          alt={feature.title}
                          width={200}
                          height={100}
                          className="w-full h-auto opacity-70 group-hover:opacity-100 transition-opacity duration-500"
                        />
                      </motion.div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </motion.div>
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="py-32 bg-gradient-to-b from-black to-gray-950 relative" ref={processRef}>
        {/* Animated lines with parallax */}
        <div className="absolute inset-0 overflow-hidden">
          {[...Array(10)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute h-px bg-amber-600/10"
              style={{
                top: `${10 + i * 8}%`,
                left: 0,
                right: 0,
                y: mousePosition.y * -10 * ((i % 3) + 1),
              }}
              animate={{
                x: ["-100%", "100%"],
              }}
              transition={{
                duration: 15 + i * 2,
                repeat: Number.POSITIVE_INFINITY,
                ease: "linear",
                delay: i * 0.5,
              }}
            />
          ))}
        </div>

        {/* Futuristic circuit board pattern with parallax */}
        <motion.div
          className="absolute inset-0 z-0 opacity-5"
          style={{
            x: mousePosition.x * -15,
            y: mousePosition.y * -15,
          }}
        >
          <svg width="100%" height="100%" viewBox="0 0 100 100" preserveAspectRatio="none">
            <pattern
              id="circuitPattern"
              patternUnits="userSpaceOnUse"
              width="100"
              height="100"
              patternTransform="scale(0.5)"
            >
              <path d="M10,10 L90,10 L90,90 L10,90 Z" fill="none" stroke="rgba(217, 119, 6, 0.8)" strokeWidth="0.5" />
              <path d="M30,10 L30,30 L10,30" fill="none" stroke="rgba(217, 119, 6, 0.8)" strokeWidth="0.5" />
              <path d="M70,10 L70,30 L90,30" fill="none" stroke="rgba(217, 119, 6, 0.8)" strokeWidth="0.5" />
              <path d="M30,90 L30,70 L10,70" fill="none" stroke="rgba(217, 119, 6, 0.8)" strokeWidth="0.5" />
              <path d="M70,90 L70,70 L90,70" fill="none" stroke="rgba(217, 119, 6, 0.8)" strokeWidth="0.5" />
              <circle cx="50" cy="50" r="5" fill="none" stroke="rgba(217, 119, 6, 0.8)" strokeWidth="0.5" />
              <path d="M50,45 L50,10" fill="none" stroke="rgba(217, 119, 6, 0.8)" strokeWidth="0.5" />
              <path d="M50,55 L50,90" fill="none" stroke="rgba(217, 119, 6, 0.8)" strokeWidth="0.5" />
              <path d="M45,50 L10,50" fill="none" stroke="rgba(217, 119, 6, 0.8)" strokeWidth="0.5" />
              <path d="M55,50 L90,50" fill="none" stroke="rgba(217, 119, 6, 0.8)" strokeWidth="0.5" />
            </pattern>
            <rect width="100%" height="100%" fill="url(#circuitPattern)" />
          </svg>
        </motion.div>

        <div className="container px-4 md:px-6 relative z-10">
          <motion.div
            className="text-center mb-24"
            variants={fadeInUpVariants}
            initial="hidden"
            animate={processInView ? "visible" : "hidden"}
            custom={0}
          >
            <h2 className="text-3xl md:text-5xl font-light mb-6 bg-clip-text text-transparent bg-gradient-to-r from-amber-400 to-yellow-600 tracking-wider">
              THE PROCESS
            </h2>
            <motion.div
              className="w-24 h-1 bg-gradient-to-r from-amber-400 to-yellow-600 mx-auto mb-6"
              initial={{ width: 0 }}
              animate={processInView ? { width: 96 } : { width: 0 }}
              transition={{ duration: 1, delay: 0.3, ease: [0.22, 1, 0.36, 1] }}
            ></motion.div>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto font-light tracking-wide">
              A sophisticated approach to video enhancement
            </p>
          </motion.div>

          <motion.div
            className="max-w-4xl mx-auto"
            variants={staggerChildrenVariants}
            initial="hidden"
            animate={processInView ? "visible" : "hidden"}
          >
            <div className="space-y-20">
              {[
                {
                  number: "01",
                  title: "FRAME ANALYSIS",
                  description:
                    "Decadence analyzes each frame of your video to understand motion patterns and object relationships.",
                },
                {
                  number: "02",
                  title: "AI PREDICTION",
                  description:
                    "The neural network predicts how objects move between frames and generates new intermediate frames.",
                },
                {
                  number: "03",
                  title: "QUALITY ENHANCEMENT",
                  description:
                    "Advanced algorithms refine the generated frames to ensure consistency and visual quality.",
                },
                {
                  number: "04",
                  title: "OUTPUT GENERATION",
                  description: "The final video is assembled with the new frames integrated for smooth, fluid motion.",
                },
              ].map((step, index) => (
                <motion.div
                  key={index}
                  variants={fadeInLeftVariants}
                  custom={index * 0.2}
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
                      className="w-16 h-16 rounded-full bg-amber-900/20 flex items-center justify-center text-amber-400 font-bold text-2xl border border-amber-600/20"
                      animate={{
                        boxShadow: [
                          "0 0 0 rgba(217, 119, 6, 0)",
                          "0 0 20px rgba(217, 119, 6, 0.3)",
                          "0 0 0 rgba(217, 119, 6, 0)",
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
                    <h3 className="text-2xl font-light text-white mb-4 tracking-wide">{step.title}</h3>
                    <p className="text-gray-400 text-lg tracking-wide">{step.description}</p>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </section>

      {/* Fix the transition between sections with an enhanced gradient divider */}
      <div className="h-32 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-gray-950 to-black"></div>
        <motion.div
          className="absolute inset-0 opacity-10"
          style={{
            backgroundImage:
              "linear-gradient(to right, rgba(217, 119, 6, 0.3) 1px, transparent 1px), linear-gradient(to bottom, rgba(217, 119, 6, 0.3) 1px, transparent 1px)",
            backgroundSize: "20px 20px",
            x: mousePosition.x * -10,
            y: mousePosition.y * -10,
          }}
        />

        {/* Add floating particles in the transition */}
        <div className="absolute inset-0 overflow-hidden">
          {[...Array(10)].map((_, i) => (
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
                duration: Math.random() * 5 + 5,
                repeat: Number.POSITIVE_INFINITY,
                ease: "easeInOut",
                delay: Math.random() * 2,
              }}
            />
          ))}
        </div>
      </div>

      {/* Download Section */}
      <section id="download" className="py-32 bg-black relative" ref={downloadRef}>
        {/* Simple background with parallax */}
        <motion.div
          className="absolute inset-0 z-0 opacity-20 bg-gradient-radial from-amber-900/20 via-transparent to-transparent"
          style={{
            x: mousePosition.x * -20,
            y: mousePosition.y * -20,
          }}
        />

        {/* Tech pattern background with parallax */}
        <motion.div
          className="absolute inset-0 z-0 opacity-5"
          style={{
            backgroundImage:
              "linear-gradient(45deg, rgba(217, 119, 6, 0.5) 1px, transparent 1px), linear-gradient(135deg, rgba(217, 119, 6, 0.5) 1px, transparent 1px)",
            backgroundSize: "20px 20px",
            x: mousePosition.x * -10,
            y: mousePosition.y * -10,
          }}
        />

        <div className="container px-4 md:px-6 relative z-10">
          <motion.div
            className="max-w-3xl mx-auto text-center"
            variants={fadeInUpVariants}
            initial="hidden"
            animate={downloadInView ? "visible" : "hidden"}
            custom={0}
          >
            <h2 className="text-3xl md:text-5xl font-light mb-6 bg-clip-text text-transparent bg-gradient-to-r from-amber-400 to-yellow-600 tracking-wider">
              DOWNLOAD DECADENCE
            </h2>
            <motion.div
              className="w-24 h-1 bg-gradient-to-r from-amber-400 to-yellow-600 mx-auto mb-8"
              initial={{ width: 0 }}
              animate={downloadInView ? { width: 96 } : { width: 0 }}
              transition={{ duration: 1, delay: 0.3, ease: [0.22, 1, 0.36, 1] }}
            ></motion.div>
            <p className="text-xl text-gray-400 mb-16 font-light tracking-wide">
              Transform your videos with professional-grade frame interpolation
            </p>

            <div className="mb-16 max-w-md mx-auto">
              <motion.div
                variants={fadeInUpVariants}
                custom={0.4}
                whileHover={{
                  y: -10,
                  boxShadow: "0 20px 25px -5px rgba(217, 119, 6, 0.1), 0 10px 10px -5px rgba(217, 119, 6, 0.04)",
                  transition: { type: "spring", stiffness: 300, damping: 15 },
                }}
                className="border border-amber-600/30 rounded-lg p-6 bg-black/50 backdrop-blur-sm hover:border-amber-500/80 hover:bg-amber-900/10 transition-all duration-500"
              >
                <h3 className="text-2xl text-white tracking-wider font-light mb-4">WINDOWS</h3>

                <div className="flex justify-between text-sm text-gray-400 tracking-wide mb-6">
                  <span>Version 1.2.0</span>
                  <span>64-bit</span>
                </div>

                <motion.div
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  transition={{ type: "spring", stiffness: 400, damping: 17 }}
                >
                  <Button className="w-full bg-gradient-to-r from-amber-600 to-yellow-600 hover:from-amber-500 hover:to-yellow-500 py-6 text-lg tracking-wider">
                    <Download className="mr-3 h-5 w-5" />
                    DOWNLOAD .EXE
                  </Button>
                </motion.div>
              </motion.div>
            </div>

            <motion.div className="flex justify-center" variants={fadeInUpVariants} custom={0.6}>
              <motion.div
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                transition={{ type: "spring", stiffness: 400, damping: 17 }}
              >
                <Button
                  variant="outline"
                  size="lg"
                  className="border-amber-600/50 text-amber-400 hover:bg-amber-600/20 hover:border-amber-500 px-8 py-6 text-lg tracking-wider"
                >
                  <Github className="mr-3 h-5 w-5" />
                  VIEW ON GITHUB
                </Button>
              </motion.div>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-16 bg-gray-950 text-gray-400 relative overflow-hidden">
        {/* Animated background lines */}
        <div className="absolute inset-0">
          {[...Array(5)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute h-px bg-amber-600/5"
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

        {/* Tech grid pattern with parallax */}
        <motion.div
          className="absolute inset-0 z-0 opacity-5"
          style={{
            backgroundImage:
              "linear-gradient(to right, rgba(217, 119, 6, 0.5) 1px, transparent 1px), linear-gradient(to bottom, rgba(217, 119, 6, 0.5) 1px, transparent 1px)",
            backgroundSize: "20px 20px",
            x: mousePosition.x * -5,
            y: mousePosition.y * -5,
          }}
        />

        <div className="container px-4 md:px-6 relative z-10">
          <motion.div
            className="flex flex-col items-center justify-center mb-12"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            <motion.div
              whileHover={{
                scale: 1.1,
                rotate: [0, -5, 5, 0],
                transition: { duration: 0.5 },
              }}
            >
              <Image src="/images/logo_no_bg.ico" alt="Decadence Logo" width={60} height={60} className="mb-6" />
            </motion.div>

            <h3 className="text-2xl font-light text-white mb-4 tracking-wider">DECADENCE</h3>
            <p className="text-gray-400 text-center max-w-md mb-8 tracking-wide">
              Advanced frame interpolation for video enhancement
            </p>
            <div className="flex space-x-6">
              {[
                <Github className="h-6 w-6" key="github" />,
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="h-6 w-6"
                  key="twitter"
                >
                  <path d="M22 4s-.7 2.1-2 3.4c1.6 10-9.4 17.3-18 11.6 2.2.1 4.4-.6 6-2C3 15.5.5 9.6 3 5c2.2 2.6 5.6 4.1 9 4-.9-4.2 4-6.6 7-3.8 1.1 0 3-1.2 3-1.2z"></path>
                </svg>,
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="24"
                  height="24"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  className="h-6 w-6"
                  key="instagram"
                >
                  <rect width="20" height="20" x="2" y="2" rx="5" ry="5"></rect>
                  <path d="M16 11.37A4 4 0 1 1 12.63 8 4 4 0 0 1 16 11.37z"></path>
                  <line x1="17.5" x2="17.51" y1="6.5" y2="6.5"></line>
                </svg>,
              ].map((icon, i) => (
                <motion.a
                  key={i}
                  href="#"
                  className="hover:text-amber-400 transition-colors"
                  whileHover={{
                    scale: 1.2,
                    rotate: [-5, 5, 0],
                    transition: { duration: 0.3 },
                  }}
                >
                  {icon}
                </motion.a>
              ))}
            </div>
          </motion.div>

          <motion.div
            className="border-t border-gray-800 pt-8 text-center"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ duration: 0.8, delay: 0.3 }}
            viewport={{ once: true }}
          >
            <p className="tracking-wide">© {new Date().getFullYear()} DECADENCE. ALL RIGHTS RESERVED.</p>
          </motion.div>
        </div>
      </footer>
    </main>
  )
}

