"use client"

import { useRef, useEffect, useState } from "react"
import { motion } from "framer-motion"
import { useInView } from "react-intersection-observer"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Wand2, Sliders, Film, Clock } from "lucide-react"

export default function FeaturesSection() {
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

  const features = [
    {
      icon: <Wand2 className="h-8 w-8 text-gold-400" />,
      title: "AI-Powered Interpolation",
      description: "Neural network models analyze motion patterns to create natural-looking intermediate frames",
    },
    {
      icon: <Sliders className="h-8 w-8 text-gold-400" />,
      title: "Customizable Settings",
      description: "Fine-tune interpolation parameters to achieve your desired visual style and performance",
    },
  ]

  const examples = [
    {
      icon: <Film className="h-6 w-6 text-gold-400" />,
      title: "24FPS to 60FPS Conversion",
      description: "Transform standard footage into silky smooth high frame rate video",
    },
    {
      icon: <Clock className="h-6 w-6 text-gold-400" />,
      title: "Low Frame Interpolation",
      description: "Interpolate two images to make content worthy videos",
    },
  ]

  return (
    <section
      id="features"
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
        className="absolute inset-0 bg-gradient-radial from-burgundy-950/30 via-background to-background z-0"
        style={{
          transform: `translateY(${(scrollY - (containerRef.current?.offsetTop || 0)) * 0.1}px)`,
        }}
      />

      <div
        className="absolute inset-0 opacity-5 z-0"
        style={{
          backgroundImage:
            "linear-gradient(45deg, rgba(237, 168, 185, 0.1) 1px, transparent 1px), linear-gradient(135deg, rgba(237, 168, 185, 0.1) 1px, transparent 1px)",
          backgroundSize: "40px 40px",
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
          <h2 className="font-serif text-3xl md:text-5xl font-light mb-6 text-secondary">Powerful Features</h2>
          <div className="w-24 h-px bg-gold-400 mx-auto mb-8"></div>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto font-light">
            Professional-grade frame interpolation with elegant simplicity
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-16 mb-32">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 50 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.8, delay: index * 0.2 }}
              style={{
                transform: `translateY(${(scrollY - (containerRef.current?.offsetTop || 0) - 300) * -0.05 * (index + 1)}px)`,
              }}
            >
              <Card className="bg-card/50 backdrop-blur-sm border-gradient elegant-shadow h-full">
                <CardHeader>
                  <div className="mb-6 p-4 rounded-full w-fit bg-burgundy-950/50">{feature.icon}</div>
                  <CardTitle className="text-2xl font-serif font-light text-secondary">{feature.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-muted-foreground text-lg">{feature.description}</CardDescription>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={inView ? { opacity: 1 } : {}}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="text-center mb-16"
        >
          <h3 className="font-serif text-2xl md:text-3xl font-light mb-4 text-secondary">Experience the Difference</h3>
          <p className="text-muted-foreground">See how Decadence transforms your videos</p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-5xl mx-auto">
          {examples.map((example, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 0.8, delay: 0.6 + index * 0.2 }}
              style={{
                transform: `translateY(${(scrollY - (containerRef.current?.offsetTop || 0) - 600) * -0.03 * (index + 1)}px)`,
              }}
            >
              <Card className="bg-card/50 backdrop-blur-sm border border-gold-400/20 hover:border-gold-400/40 transition-colors elegant-shadow h-full overflow-hidden">
                <CardHeader className="pb-2">
                  <div className="flex items-center mb-2">
                    <div className="mr-3 p-2 rounded-full bg-burgundy-950/50">{example.icon}</div>
                    <CardTitle className="text-xl font-serif font-light text-secondary">{example.title}</CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-muted-foreground mb-4">{example.description}</CardDescription>
                  <div className="mt-4 overflow-hidden rounded-md bg-burgundy-950/30 border border-burgundy-900/20">
                    <div className="w-full h-32 bg-gradient-to-br from-burgundy-900/20 to-gold-900/20 animate-shimmer bg-[length:200%_100%]"></div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  )
}

