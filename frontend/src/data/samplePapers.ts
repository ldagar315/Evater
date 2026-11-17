export interface SamplePaper {
  id: string
  title: string
  subtitle: string
  classLevel: number
  subject: 'Science' | 'Mathematics'
  topic: string
  keywords: string[]
  href: string
  questions: number
  marks: number
  minutes: number
  format?: 'chapter' | 'mixed'
  objectiveQuestions?: number
  writtenQuestions?: number
}

/** Original chapter and mixed-chapter practice papers. Existing chapter PDF bytes are unchanged. */
export const samplePapers: SamplePaper[] = [
  {"id": "maths-mixed", "title": "Mathematics mixed-chapter practice paper", "subtitle": "Longer practice \u00b7 Short answers, extended answers and cases", "classLevel": 8, "subject": "Mathematics", "topic": "Maths mixed chapters", "keywords": ["math", "maths", "sample paper", "practice paper", "mixed", "exam", "case", "reasoning"], "href": "/class-8-maths-mixed-chapter-practice-paper.pdf", "questions": 24, "marks": 60, "minutes": 150, "format": "mixed", "objectiveQuestions": 12, "writtenQuestions": 12},
  {"id": "science-mixed", "title": "Science mixed-chapter practice paper", "subtitle": "Longer practice \u00b7 Short answers, extended answers and cases", "classLevel": 8, "subject": "Science", "topic": "Science mixed chapters", "keywords": ["science", "sample paper", "practice paper", "mixed", "exam", "case", "reasoning"], "href": "/class-8-science-mixed-chapter-practice-paper.pdf", "questions": 24, "marks": 60, "minutes": 150, "format": "mixed", "objectiveQuestions": 12, "writtenQuestions": 12},

  { id: 'light', title: 'Light', subtitle: 'Mirrors and lenses', classLevel: 8, subject: 'Science', topic: 'Light', keywords: ['reflection', 'refraction', 'optics', 'convex', 'concave', 'mirrors', 'lenses'], href: '/1-class-8-light-sample-paper.pdf', questions: 15, marks: 22, minutes: 45 },
  { id: 'electricity', title: 'Electricity', subtitle: 'Magnetic and heating effects', classLevel: 8, subject: 'Science', topic: 'Electricity', keywords: ['magnetism', 'electromagnet', 'current', 'circuits', 'heating', 'cells'], href: '/2-class-8-electricity-sample-paper.pdf', questions: 15, marks: 22, minutes: 45 },
  { id: 'pythagoras', title: 'The Pythagoras theorem', subtitle: 'The Baudhayana-Pythagoras theorem', classLevel: 8, subject: 'Mathematics', topic: 'Pythagoras', keywords: ['maths', 'math', 'right triangle', 'triangles', 'hypotenuse', 'geometry', 'Baudhayana'], href: '/3-class-8-pythagoras-sample-paper.pdf', questions: 15, marks: 22, minutes: 45 },
  {"id": "maths-ch01", "title": "A Square and A Cube", "subtitle": "Chapter 1 · Squares & cubes", "classLevel": 8, "subject": "Mathematics", "topic": "Squares & cubes", "keywords": ["math", "maths", "squares", "cubes", "square roots", "triangular"], "href": "/class-8-maths-ch01-a-square-and-a-cube-sample-paper.pdf", "questions": 15, "marks": 22, "minutes": 45},
  {"id": "maths-ch02", "title": "Power Play", "subtitle": "Chapter 2 · Powers", "classLevel": 8, "subject": "Mathematics", "topic": "Powers", "keywords": ["math", "maths", "exponents", "scientific notation", "negative powers"], "href": "/class-8-maths-ch02-power-play-sample-paper.pdf", "questions": 15, "marks": 22, "minutes": 45},
  {"id": "maths-ch03", "title": "A Story of Numbers", "subtitle": "Chapter 3 · Number systems", "classLevel": 8, "subject": "Mathematics", "topic": "Number systems", "keywords": ["math", "maths", "binary", "roman", "egyptian", "place value", "base"], "href": "/class-8-maths-ch03-a-story-of-numbers-sample-paper.pdf", "questions": 15, "marks": 22, "minutes": 45},
  {"id": "maths-ch04", "title": "Quadrilaterals", "subtitle": "Chapter 4 · Quadrilaterals", "classLevel": 8, "subject": "Mathematics", "topic": "Quadrilaterals", "keywords": ["math", "maths", "diagonals", "rhombus", "rectangle", "kite", "geometry"], "href": "/class-8-maths-ch04-quadrilaterals-sample-paper.pdf", "questions": 15, "marks": 22, "minutes": 45},
  {"id": "maths-ch05", "title": "Number Play", "subtitle": "Chapter 5 · Number patterns", "classLevel": 8, "subject": "Mathematics", "topic": "Number patterns", "keywords": ["math", "maths", "primes", "factors", "multiples", "hcf", "lcm", "divisibility"], "href": "/class-8-maths-ch05-number-play-sample-paper.pdf", "questions": 15, "marks": 22, "minutes": 45},
  {"id": "maths-ch06", "title": "We Distribute, Yet Things Multiply", "subtitle": "Chapter 6 · Algebraic identities", "classLevel": 8, "subject": "Mathematics", "topic": "Algebraic identities", "keywords": ["math", "maths", "distributive", "expansion", "factorisation", "identities"], "href": "/class-8-maths-ch06-we-distribute-yet-things-multiply-sample-paper.pdf", "questions": 15, "marks": 22, "minutes": 45},
  {"id": "maths-ch07", "title": "Proportional Reasoning-1", "subtitle": "Chapter 7 · Ratios I", "classLevel": 8, "subject": "Mathematics", "topic": "Ratios I", "keywords": ["math", "maths", "ratio", "proportion", "unitary", "scale"], "href": "/class-8-maths-ch07-proportional-reasoning-1-sample-paper.pdf", "questions": 15, "marks": 22, "minutes": 45},
  {"id": "maths-ch08", "title": "Fractions in Disguise", "subtitle": "Chapter 8 · Percentages", "classLevel": 8, "subject": "Mathematics", "topic": "Percentages", "keywords": ["math", "maths", "fraction", "percent", "discount", "profit", "loss", "interest"], "href": "/class-8-maths-ch08-fractions-in-disguise-sample-paper.pdf", "questions": 15, "marks": 22, "minutes": 45},
  {"id": "maths-ch10", "title": "Proportional Reasoning-2", "subtitle": "Chapter 10 · Ratios II", "classLevel": 8, "subject": "Mathematics", "topic": "Ratios II", "keywords": ["math", "maths", "inverse proportion", "pie chart", "map", "mixture", "rates"], "href": "/class-8-maths-ch10-proportional-reasoning-2-sample-paper.pdf", "questions": 15, "marks": 22, "minutes": 45},
  {"id": "maths-ch11", "title": "Exploring Some Geometric Themes", "subtitle": "Chapter 11 · Fractals & solids", "classLevel": 8, "subject": "Mathematics", "topic": "Fractals & solids", "keywords": ["math", "maths", "sierpinski", "koch", "net", "cube", "polyhedron", "euler"], "href": "/class-8-maths-ch11-exploring-some-geometric-themes-sample-paper.pdf", "questions": 15, "marks": 22, "minutes": 45},
  {"id": "maths-ch12", "title": "Tales by Dots and Lines", "subtitle": "Chapter 12 · Data & graphs", "classLevel": 8, "subject": "Mathematics", "topic": "Data & graphs", "keywords": ["math", "maths", "mean", "average", "line graph", "bar graph"], "href": "/class-8-maths-ch12-tales-by-dots-and-lines-sample-paper.pdf", "questions": 15, "marks": 22, "minutes": 45},
  {"id": "maths-ch13", "title": "Algebra Play", "subtitle": "Chapter 13 · Algebra puzzles", "classLevel": 8, "subject": "Mathematics", "topic": "Algebra puzzles", "keywords": ["math", "maths", "pyramid", "number trick", "calendar", "divisibility"], "href": "/class-8-maths-ch13-algebra-play-sample-paper.pdf", "questions": 15, "marks": 22, "minutes": 45},
  {"id": "maths-ch14", "title": "Area", "subtitle": "Chapter 14 · Area", "classLevel": 8, "subject": "Mathematics", "topic": "Area", "keywords": ["math", "maths", "triangle", "rhombus", "parallelogram", "trapezium", "measurement"], "href": "/class-8-maths-ch14-area-sample-paper.pdf", "questions": 15, "marks": 22, "minutes": 45},
  {"id": "science-ch01", "title": "Exploring the Investigative World of Science", "subtitle": "Chapter 1 · Scientific enquiry", "classLevel": 8, "subject": "Science", "topic": "Scientific enquiry", "keywords": ["investigation", "experiment", "variables", "fair test", "observation"], "href": "/class-8-science-ch01-exploring-the-investigative-world-of-science-sample-paper.pdf", "questions": 15, "marks": 22, "minutes": 45},
  {"id": "science-ch02", "title": "The Invisible Living World: Beyond Our Naked Eye", "subtitle": "Chapter 2 · Microorganisms", "classLevel": 8, "subject": "Science", "topic": "Microorganisms", "keywords": ["cell", "bacteria", "fungi", "virus", "microscope", "fermentation"], "href": "/class-8-science-ch02-the-invisible-living-world-beyond-our-naked-eye-sample-paper.pdf", "questions": 15, "marks": 22, "minutes": 45},
  {"id": "science-ch03", "title": "Health: The Ultimate Treasure", "subtitle": "Chapter 3 · Health", "classLevel": 8, "subject": "Science", "topic": "Health", "keywords": ["disease", "immunity", "nutrition", "hygiene", "vaccination"], "href": "/class-8-science-ch03-health-the-ultimate-treasure-sample-paper.pdf", "questions": 15, "marks": 22, "minutes": 45},
  {"id": "science-ch05", "title": "Exploring Forces", "subtitle": "Chapter 5 · Forces", "classLevel": 8, "subject": "Science", "topic": "Forces", "keywords": ["friction", "gravity", "magnetic", "electrostatic", "buoyancy", "mass", "weight"], "href": "/class-8-science-ch05-exploring-forces-sample-paper.pdf", "questions": 15, "marks": 22, "minutes": 45},
  {"id": "science-ch06", "title": "Pressure, Winds, Storms, and Cyclones", "subtitle": "Chapter 6 · Pressure & winds", "classLevel": 8, "subject": "Science", "topic": "Pressure & winds", "keywords": ["pressure", "cyclone", "storm", "lightning", "wind", "atmosphere"], "href": "/class-8-science-ch06-pressure-winds-storms-and-cyclones-sample-paper.pdf", "questions": 15, "marks": 22, "minutes": 45},
  {"id": "science-ch07", "title": "Particulate Nature of Matter", "subtitle": "Chapter 7 · Particles", "classLevel": 8, "subject": "Science", "topic": "Particles", "keywords": ["matter", "solid", "liquid", "gas", "diffusion", "evaporation"], "href": "/class-8-science-ch07-particulate-nature-of-matter-sample-paper.pdf", "questions": 15, "marks": 22, "minutes": 45},
  {"id": "science-ch08", "title": "Nature of Matter: Elements, Compounds, and Mixtures", "subtitle": "Chapter 8 · Elements & mixtures", "classLevel": 8, "subject": "Science", "topic": "Elements & mixtures", "keywords": ["element", "compound", "mixture", "alloy", "separation"], "href": "/class-8-science-ch08-nature-of-matter-elements-compounds-and-mixtures-sample-paper.pdf", "questions": 15, "marks": 22, "minutes": 45},
  {"id": "science-ch09", "title": "The Amazing World of Solutes, Solvents, and Solutions", "subtitle": "Chapter 9 · Solutions", "classLevel": 8, "subject": "Science", "topic": "Solutions", "keywords": ["solute", "solvent", "solution", "saturation", "density"], "href": "/class-8-science-ch09-the-amazing-world-of-solutes-solvents-and-solutions-sample-paper.pdf", "questions": 15, "marks": 22, "minutes": 45},
  {"id": "science-ch11", "title": "Keeping Time with the Skies", "subtitle": "Chapter 11 · Moon & calendars", "classLevel": 8, "subject": "Science", "topic": "Moon & calendars", "keywords": ["moon", "phases", "calendar", "leap", "solar", "lunar"], "href": "/class-8-science-ch11-keeping-time-with-the-skies-sample-paper.pdf", "questions": 15, "marks": 22, "minutes": 45},
  {"id": "science-ch12", "title": "How Nature Works in Harmony", "subtitle": "Chapter 12 · Ecosystems", "classLevel": 8, "subject": "Science", "topic": "Ecosystems", "keywords": ["habitat", "food chain", "food web", "biodiversity", "decomposer"], "href": "/class-8-science-ch12-how-nature-works-in-harmony-sample-paper.pdf", "questions": 15, "marks": 22, "minutes": 45},
  {"id": "science-ch13", "title": "Our Home: Earth, a Unique Life Sustaining Planet", "subtitle": "Chapter 13 · Earth & life", "classLevel": 8, "subject": "Science", "topic": "Earth & life", "keywords": ["earth systems", "life", "reproduction", "climate", "biodiversity", "geodiversity"], "href": "/class-8-science-ch13-our-home-earth-a-unique-life-sustaining-planet-sample-paper.pdf", "questions": 15, "marks": 22, "minutes": 45},

]

export interface PaperFilters { query: string; classLevel: string; subject: string; topic: string }
export function filterPapers(papers: SamplePaper[], filters: PaperFilters) {
  const words = filters.query.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean)
  return papers.filter(paper => {
    const text = [paper.title, paper.subtitle, paper.topic, paper.subject, `class ${paper.classLevel}`, ...paper.keywords].join(' ').toLocaleLowerCase()
    return (!filters.classLevel || String(paper.classLevel) === filters.classLevel)
      && (!filters.subject || paper.subject === filters.subject)
      && (!filters.topic || paper.topic === filters.topic)
      && words.every(word => text.includes(word))
  })
}
