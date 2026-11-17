export type PaperCollectionKey = 'all' | 'maths' | 'science'
interface PaperCollection { path: string; label: string; title: string; description: string; heading: string; intro: string; coverage: string; shelf: string; listHeading: string; subject?: 'Mathematics' | 'Science' }
export const paperCollections: Record<PaperCollectionKey, PaperCollection> = {
  all: {
    path: '/papers/', label: 'All Class 8 papers', title: 'Class 8 Sample Papers: Maths & Science PDFs with Answers | Evater',
    description: 'Download 27 original Class 8 Maths and Science chapter sample papers. Printable PDFs with answers and marking guidance. No sign-in needed.',
    heading: 'Class 8 sample papers',
    intro: 'Choose a Maths or Science chapter, print its practice paper and check your answers after you finish.',
    coverage: 'Browse 14 Mathematics and 13 Science chapter papers, covering topics from Ganita Prakash and Curiosity. Find a paper by chapter name or by a topic such as percentages, forces or light.',
    shelf: 'Class 8 Science and Mathematics chapter papers.', listHeading: 'Browse all 27 chapter papers',
  },
  maths: {
    path: '/papers/class-8-maths/', label: 'Class 8 Maths', title: 'Class 8 Maths Sample Papers with Answers: Chapter PDFs | Evater',
    description: 'Practise Class 8 Maths with 14 original chapter sample papers. Download PDFs with worked answers and marking guidance, from squares and cubes to area.',
    heading: 'Class 8 Maths sample papers',
    intro: 'Build confidence one chapter at a time. Download a printable Maths practice paper, show your working, then compare it with the model answers.',
    coverage: 'These 14 papers cover topics from Ganita Prakash: squares and cubes, powers, number systems, quadrilaterals, divisibility, algebraic identities, ratios, percentages, the Pythagoras theorem, geometric themes, data and graphs, algebra puzzles and area.',
    shelf: 'Class 8 Mathematics: 14 chapter papers.', listHeading: 'Browse 14 Maths chapter papers', subject: 'Mathematics',
  },
  science: {
    path: '/papers/class-8-science/', label: 'Class 8 Science', title: 'Class 8 Science Sample Papers with Answers: Chapter PDFs | Evater',
    description: 'Download 13 original Class 8 Science chapter sample papers with answers. Printable Curiosity-topic practice PDFs for investigation, forces, matter, light and more.',
    heading: 'Class 8 Science sample papers',
    intro: 'Check what you understand, not just what you remember. Choose a Science chapter and try its objective and written-response questions before reading the answers.',
    coverage: 'These 13 papers cover topics from Curiosity: scientific investigation, microorganisms, health, electricity, forces, pressure and winds, particles, elements and mixtures, solutions, light, skies and calendars, ecosystems, and Earth as a life-sustaining planet.',
    shelf: 'Class 8 Science: 13 chapter papers.', listHeading: 'Browse 13 Science chapter papers', subject: 'Science',
  },
}
