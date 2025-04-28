"use client"

import { useRef } from "react"
import { motion, useScroll, useTransform } from "framer-motion"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Wand2, Sliders, Film, Clock } from "lucide-react"

export default function Features() {
  const containerRef = useRef<HTMLDivElement>(null)
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end start"],
  })

  const y1 = useTransform(scrollYProgress, [0, 1], [0, -100])
  const y2 = useTransform(scrollYProgress, [0, 1], [0, -200])
  const opacity = useTransform(scrollYProgress, [0, 0.2, 0.9, 1], [0, 1, 1, 0])

  const features = [
    {
      icon: <Wand2 className="h-8 w-8 text-secondary" />,
      title: "AI-Powered Interpolation",
      description: "Neural network models analyze motion patterns to create natural-looking intermediate frames",
    },
    {
      icon: <Sliders className="h-8 w-8 text-secondary" />,
      title: "Customizable Settings",
      description: "Fine-tune interpolation parameters to achieve your desired visual style and performance",
    },
  ]

  const examples = [
    {
      icon: <Film className="h-6 w-6 text-secondary" />,
      title: "24FPS to 60FPS Conversion",
      description: "Transform standard footage into silky smooth high frame rate video",
    },
    {
      icon: <Clock className="h-6 w-6 text-secondary" />,
      title: "Low Frame Interpolation",
      description: "Interpolate two images to make content worthy videos",
    },
  ]

  return (
    <section id="features" ref={containerRef} className="relative py-32 overflow-hidden">
      {/* Background elements with parallax */}
      <motion.div
        className="absolute inset-0 bg-gradient-radial from-purple-900/20 via-background to-background z-0"
        style={{ y: y1 }}
      />

      <motion.div className="absolute inset-0 hexagon-grid opacity-30 z-0" style={{ y: y2 }} />

      <div className="container relative z-10 px-4 md:px-6">
        <motion.div
          initial={{ opacity: 0, y: 50 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true, margin: "-100px" }}
          className="text-center mb-24"
        >
          <h2 className="font-serif text-4xl md:text-6xl font-light mb-6 gradient-text glow-text tracking-wide">
            POWERFUL FEATURES
          </h2>
          <div className="w-24 h-px bg-secondary mx-auto mb-8"></div>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto font-light tracking-wide">
            Professional-grade frame interpolation with elegant simplicity
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-16 mb-32">
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 50 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: index * 0.2 }}
              viewport={{ once: true, margin: "-100px" }}
              whileHover={{ y: -10 }}
            >
              <Card className="bg-card/50 backdrop-blur-sm gradient-border glow h-full">
                <CardHeader>
                  <div className="mb-6 p-4 rounded-full w-fit bg-primary/20">{feature.icon}</div>
                  <CardTitle className="text-2xl font-serif font-light text-secondary tracking-wide">
                    {feature.title}
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-muted-foreground text-lg tracking-wide">
                    {feature.description}
                  </CardDescription>
                </CardContent>
              </Card>
            </motion.div>
          ))}
        </div>

        <motion.div
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          viewport={{ once: true, margin: "-100px" }}
          className="text-center mb-16"
        >
          <h3 className="font-serif text-3xl md:text-4xl font-light mb-4 gradient-text tracking-wide">
            EXPERIENCE THE DIFFERENCE
          </h3>
          <p className="text-muted-foreground tracking-wide">See how Decadence transforms your videos</p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-5xl mx-auto">
          {examples.map((example, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.6 + index * 0.2 }}
              viewport={{ once: true, margin: "-100px" }}
              whileHover={{ y: -5 }}
            >
              <Card className="bg-card/50 backdrop-blur-sm border border-secondary/20 hover:border-secondary/40 transition-colors glow h-full overflow-hidden">
                <CardHeader className="pb-2">
                  <div className="flex items-center mb-2">
                    <div className="mr-3 p-2 rounded-full bg-primary/20">{example.icon}</div>
                    <CardTitle className="text-xl font-serif font-light text-secondary tracking-wide">
                      {example.title}
                    </CardTitle>
                  </div>
                </CardHeader>
                <CardContent>
                  <CardDescription className="text-muted-foreground mb-4 tracking-wide">
                    {example.description}
                  </CardDescription>
                  <div className="mt-4 overflow-hidden rounded-md bg-primary/10 border border-primary/20">
                    <div className="w-full h-32 bg-gradient-to-br from-primary/20 to-secondary/20 animate-shimmer bg-[length:200%_100%]"></div>
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

