import { Brain, Layers3, Zap } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

interface ModelSelectorProps {
  selectedModel: 'cnn-2d' | 'cnn-3d' | 'hybrid'
  onModelChange: (model: 'cnn-2d' | 'cnn-3d' | 'hybrid') => void
}

const models = [
  {
    id: 'cnn-2d' as const,
    name: 'CNN 2D',
    description: 'Réseau de neurones convolutif 2D pour l\'analyse de pseudo-sections',
    icon: Brain,
    accuracy: 94.2,
    speed: 'Rapide',
    useCase: 'Pseudo-sections de résistivité'
  },
  {
    id: 'cnn-3d' as const,
    name: 'CNN 3D',
    description: 'Modèle 3D pour la visualisation volumique du sous-sol',
    icon: Layers3,
    accuracy: 91.8,
    speed: 'Moyen',
    useCase: 'Modèles 3D de chargeabilité'
  },
  {
    id: 'hybrid' as const,
    name: 'Modèle Hybride',
    description: 'Combinaison d\'images et de données géophysiques',
    icon: Zap,
    accuracy: 96.1,
    speed: 'Lent',
    useCase: 'Analyse multi-modale'
  }
]

export function ModelSelector({ selectedModel, onModelChange }: ModelSelectorProps) {
  return (
    <div className="space-y-2 sm:space-y-3">
      <label className="text-sm font-medium">Modèle d'IA</label>
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-1 gap-2">
        {models.map((model) => {
          const Icon = model.icon
          return (
            <Card 
              key={model.id}
              className={`cursor-pointer transition-all hover:shadow-md ${
                selectedModel === model.id 
                  ? 'ring-2 ring-primary bg-primary/5' 
                  : 'hover:bg-muted/50'
              }`}
              onClick={() => onModelChange(model.id)}
            >
              <CardHeader className="pb-2 p-3 sm:p-4">
                <div className="flex items-start space-x-2 sm:space-x-3">
                  <Icon className="w-4 h-4 sm:w-5 sm:h-5 text-primary flex-shrink-0 mt-0.5" />
                  <div className="flex-1 min-w-0">
                    <CardTitle className="text-xs sm:text-sm truncate">{model.name}</CardTitle>
                    <CardDescription className="text-xs hidden sm:block">
                      {model.description}
                    </CardDescription>
                  </div>
                  <div className="flex flex-col items-end space-y-1 flex-shrink-0">
                    <Badge variant="secondary" className="text-xs">
                      {model.accuracy}%
                    </Badge>
                    <Badge variant="outline" className="text-xs hidden sm:inline-flex">
                      {model.speed}
                    </Badge>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="pt-0 p-3 sm:p-4 sm:pt-0">
                <p className="text-xs text-muted-foreground">
                  <span className="hidden sm:inline"><strong>Usage:</strong> </span>{model.useCase}
                </p>
              </CardContent>
            </Card>
          )
        })}
      </div>
    </div>
  )
}
