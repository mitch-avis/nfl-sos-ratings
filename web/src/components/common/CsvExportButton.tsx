import { Download } from 'lucide-react'

import { Button } from '@/components/ui/button'

/** Offer `text` as a CSV file named `fileName`, through a temporary object URL. */
function downloadCsv(fileName: string, text: string) {
  const url = URL.createObjectURL(new Blob([text], { type: 'text/csv;charset=utf-8' }))
  const link = document.createElement('a')
  link.href = url
  link.download = fileName
  link.click()
  URL.revokeObjectURL(url)
}

/** A small button that downloads the CSV `build` returns when pressed. */
export function CsvExportButton({ fileName, build }: { fileName: string; build: () => string }) {
  return (
    <Button
      variant="outline"
      size="sm"
      aria-label="Export the table as CSV"
      onClick={() => downloadCsv(fileName, build())}
    >
      <Download />
      CSV
    </Button>
  )
}
