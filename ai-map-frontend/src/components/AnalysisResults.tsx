import { } from 'react'
import { BarChart3, Layers3, Download, TrendingUp, Clock, Database } from 'lucide-react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { AnalysisResult, AnalysisProgress } from '@/types'

interface AnalysisResultsProps {
  result: AnalysisResult | null
  isAnalyzing: boolean
  progress: AnalysisProgress | null
}

export function AnalysisResults({ result, isAnalyzing, progress }: AnalysisResultsProps) {
  console.log('AnalysisResults render:', { result: !!result, isAnalyzing, progress: !!progress })
  
  if (!result && !isAnalyzing) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Layers3 className="w-5 h-5 text-primary" />
            <span>Résultats d'Analyse</span>
          </CardTitle>
          <CardDescription>Visualisations 2D/3D et modèles générés automatiquement</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center h-96 text-center">
            <div className="space-y-4">
              <div className="w-16 h-16 bg-muted rounded-full flex items-center justify-center mx-auto">
                <BarChart3 className="w-8 h-8 text-muted-foreground" />
              </div>
              <div>
                <h3 className="text-lg font-medium text-foreground mb-2">Prêt pour l'analyse</h3>
                <p className="text-muted-foreground max-w-md">
                  Sélectionnez une méthode de mesure et téléchargez vos données pour commencer l'analyse
                  géophysique automatisée.
                </p>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (isAnalyzing && progress) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle className="flex items-center space-x-2">
            <Layers3 className="w-5 h-5 text-primary" />
            <span>Analyse en cours</span>
          </CardTitle>
          <CardDescription>{progress.message}</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span>Progression</span>
                <span>{progress.progress}%</span>
              </div>
              <Progress value={progress.progress} className="w-full" />
            </div>
            
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-4 bg-muted rounded-lg">
                <div className="text-2xl font-bold text-primary">{progress.progress}%</div>
                <div className="text-sm text-muted-foreground">Complété</div>
              </div>
              <div className="text-center p-4 bg-muted rounded-lg">
                <div className="text-2xl font-bold text-accent">
                  {progress.estimatedTime ? `${progress.estimatedTime}s` : '--'}
                </div>
                <div className="text-sm text-muted-foreground">Temps restant</div>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (result) {
    return (
      <Card className="h-full">
        <CardHeader>
          <CardTitle className="flex items-center justify-between">
            <span className="flex items-center space-x-2">
              <Layers3 className="w-5 h-5 text-primary" />
              <span>Résultats d'Analyse</span>
            </span>
            <Button variant="outline" size="sm">
              <Download className="w-4 h-4 mr-2" />
              Exporter
            </Button>
          </CardTitle>
          <CardDescription>Visualisations 2D/3D et modèles générés automatiquement</CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs defaultValue="2d" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="2d">Pseudo-sections 2D</TabsTrigger>
              <TabsTrigger value="3d">Modèles 3D</TabsTrigger>
              <TabsTrigger value="data">Données</TabsTrigger>
            </TabsList>

            <TabsContent value="2d" className="mt-6">
              <div className="space-y-4">
                <div className="bg-muted rounded-lg h-64 flex items-center justify-center overflow-hidden">
                  {result.visualizations.pseudoSection2D ? (
                    <img 
                      src={result.visualizations.pseudoSection2D} 
                      alt="Pseudo-section de résistivité"
                      className="w-full h-full object-contain rounded-lg"
                    />
                  ) : (
                    <div className="text-center">
                      <BarChart3 className="w-12 h-12 text-primary mx-auto mb-2" />
                      <p className="text-sm text-muted-foreground">Pseudo-section de résistivité générée</p>
                      <Badge variant="secondary" className="mt-2">
                        {result.method.toUpperCase()}
                      </Badge>
                    </div>
                  )}
                </div>
                <div className="bg-muted rounded-lg h-64 flex items-center justify-center overflow-hidden">
                  {result.visualizations.chargeabilityMap ? (
                    <img 
                      src={result.visualizations.chargeabilityMap} 
                      alt="Carte de chargeabilité"
                      className="w-full h-full object-contain rounded-lg"
                    />
                  ) : (
                    <div className="text-center">
                      <BarChart3 className="w-12 h-12 text-accent mx-auto mb-2" />
                      <p className="text-sm text-muted-foreground">Carte d'iso-chargeabilité</p>
                      <Badge variant="outline" className="mt-2">
                        Modèle {result.method}
                      </Badge>
                    </div>
                  )}
                </div>
              </div>
            </TabsContent>

            <TabsContent value="3d" className="mt-6">
              <div className="bg-muted rounded-lg h-96 flex items-center justify-center overflow-hidden">
                {result.visualizations.model3D ? (
                  <img 
                    src={result.visualizations.model3D} 
                    alt="Modèle 3D"
                    className="w-full h-full object-contain rounded-lg"
                  />
                ) : (
                  <div className="text-center">
                    <Layers3 className="w-16 h-16 text-primary mx-auto mb-4" />
                    <h3 className="text-lg font-medium mb-2">Modèle 3D Interactif</h3>
                    <p className="text-sm text-muted-foreground mb-4">
                      Visualisation 3D de la chargeabilité du sous-sol
                    </p>
                    <Badge variant="secondary">
                      Précision: {result.metrics.accuracy}%
                    </Badge>
                  </div>
                )}
              </div>
            </TabsContent>

            <TabsContent value="data" className="mt-6">
              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm flex items-center space-x-2">
                        <TrendingUp className="w-4 h-4" />
                        <span>Précision du modèle</span>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold text-accent">{result.metrics.accuracy}%</div>
                      <p className="text-xs text-muted-foreground">+2.1% vs méthodes classiques</p>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader className="pb-2">
                      <CardTitle className="text-sm flex items-center space-x-2">
                        <Clock className="w-4 h-4" />
                        <span>Temps de traitement</span>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold text-primary">{result.metrics.processingTime} min</div>
                      <p className="text-xs text-muted-foreground">-78% vs inversion manuelle</p>
                    </CardContent>
                  </Card>
                </div>
                
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm flex items-center space-x-2">
                      <Database className="w-4 h-4" />
                      <span>Informations sur les données</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <span className="text-muted-foreground">Points de données:</span>
                        <span className="ml-2 font-medium">{result.metrics.dataPoints.toFixed(0)}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Méthode:</span>
                        <span className="ml-2 font-medium">{result.method}</span>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Statut:</span>
                        <Badge 
                          variant={result.status === 'success' ? 'default' : 'destructive'}
                          className="ml-2"
                        >
                          {result.status}
                        </Badge>
                      </div>
                      <div>
                        <span className="text-muted-foreground">Timestamp:</span>
                        <span className="ml-2 font-medium">
                          {result.timestamp.toLocaleTimeString()}
                        </span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    )
  }

  return null
}
