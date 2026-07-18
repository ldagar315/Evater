import process from 'node:process'
import { runSupabase } from './stage-env.mjs'

runSupabase(process.argv.slice(2))
