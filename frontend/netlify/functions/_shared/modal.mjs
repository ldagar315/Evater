const DEFAULT_MODAL_API_URL = 'https://ldagar315--evater-v1-wrapper.modal.run'

export const modalApiUrl = (process.env.MODAL_API_URL || DEFAULT_MODAL_API_URL).replace(/\/+$/, '')
